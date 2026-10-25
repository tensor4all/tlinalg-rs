//! The `tlbench` campaign harness.

pub mod cases;
pub mod rows;
pub mod timing;
pub mod verify;

use crate::Env;
use cases::{parse_shape, parse_usizes, Case, FAMILIES};
use std::num::NonZeroUsize;
use std::path::Path;
use std::process::ExitCode;

const USAGE: &str = "usage: tlbench info | verify --threads N --n LIST [--batch LIST] [--dtype LIST] [--family LIST] [--regime LABEL] [--rows ROWS] [--rotate-rows K] [--csv PATH] | run --threads N --n LIST [--batch LIST] [--dtype LIST] [--family LIST] [--regime LABEL] [--rows ROWS] [--rotate-rows K] [--reps M] [--prime-ms MS] [--statistic best|median] [--csv PATH]";

#[derive(Clone, Copy)]
enum Dtype {
    F64,
    C64,
}
impl Dtype {
    fn label(self) -> &'static str {
        match self {
            Self::F64 => "f64",
            Self::C64 => "c64",
        }
    }
}

struct Args {
    command: String,
    threads: Option<usize>,
    shapes: Vec<(usize, usize)>,
    batches: Vec<usize>,
    dtypes: Vec<Dtype>,
    families: Vec<String>,
    regime: String,
    reps: usize,
    prime_ms: u64,
    csv: Option<String>,
    /// Provider rows this run emits, in the order the campaign declares them. Empty means every row.
    rows: Vec<String>,
    /// Which call of a row's chronological samples the row reports.
    statistic: timing::Statistic,
    /// Cyclic rotation of `rows`, applied per case, so a comparison is position-balanced.
    rotate_rows: usize,
}

fn parse(args: &[String]) -> Result<Args, String> {
    if args.len() < 2 {
        return Err(USAGE.into());
    }
    let command = args[1].clone();
    if command == "info" {
        if args.len() != 2 {
            return Err(USAGE.into());
        }
        return Ok(Args {
            command,
            threads: None,
            shapes: vec![],
            batches: vec![],
            dtypes: vec![],
            families: vec![],
            regime: "unspecified".into(),
            reps: 5,
            prime_ms: 500,
            csv: None,
            rows: Vec::new(),
            statistic: timing::Statistic::Best,
            rotate_rows: 0,
        });
    }
    if command != "run" && command != "verify" {
        return Err(USAGE.into());
    }
    let mut a = Args {
        command,
        threads: None,
        shapes: vec![],
        batches: vec![1],
        dtypes: vec![Dtype::F64],
        families: FAMILIES.iter().map(|s| (*s).into()).collect(),
        regime: "unspecified".into(),
        reps: 5,
        prime_ms: 500,
        csv: None,
        rows: Vec::new(),
        statistic: timing::Statistic::Best,
        rotate_rows: 0,
    };
    let mut i = 2;
    while i < args.len() {
        let flag = &args[i];
        let value = |i: &mut usize| -> Result<String, String> {
            *i += 1;
            args.get(*i).cloned().ok_or_else(|| USAGE.into())
        };
        match flag.as_str() {
            "--threads" => a.threads = Some(value(&mut i)?.parse().map_err(|_| USAGE.to_owned())?),
            "--n" => {
                a.shapes = value(&mut i)?
                    .split(',')
                    .map(parse_shape)
                    .collect::<Result<_, _>>()?
            }
            "--batch" => a.batches = parse_usizes(&value(&mut i)?, "batch")?,
            "--dtype" => {
                a.dtypes = value(&mut i)?
                    .split(',')
                    .map(|d| match d {
                        "f64" => Ok(Dtype::F64),
                        "c64" => Ok(Dtype::C64),
                        "f32" | "c32" => {
                            Err("f32/c32 do not satisfy BenchScalar in this crate".into())
                        }
                        _ => Err(format!("unknown dtype {d}")),
                    })
                    .collect::<Result<_, String>>()?
            }
            "--family" => {
                let v = value(&mut i)?;
                a.families = v.split(',').map(str::to_owned).collect();
                if a.families.iter().any(|f| !FAMILIES.contains(&f.as_str())) {
                    return Err("unknown family".into());
                }
            }
            "--regime" => a.regime = value(&mut i)?,
            "--reps" => a.reps = value(&mut i)?.parse().map_err(|_| USAGE.to_owned())?,
            "--prime-ms" => a.prime_ms = value(&mut i)?.parse().map_err(|_| USAGE.to_owned())?,
            "--rows" => {
                a.rows = value(&mut i)?.split(',').map(str::to_owned).collect();
                a.rows.retain(|r| !r.is_empty());
            }
            "--rotate-rows" => {
                a.rotate_rows = value(&mut i)?.parse().map_err(|_| USAGE.to_owned())?
            }
            "--statistic" => {
                a.statistic = match value(&mut i)?.as_str() {
                    "best" => timing::Statistic::Best,
                    "median" => timing::Statistic::Median,
                    _ => return Err("--statistic must be best or median".into()),
                }
            }
            "--csv" => a.csv = Some(value(&mut i)?),
            _ => return Err(USAGE.into()),
        }
        i += 1;
    }
    if a.threads.is_none() || a.shapes.is_empty() {
        return Err(USAGE.into());
    }
    if a.threads == Some(0) {
        return Err("--threads must be positive".into());
    }
    Ok(a)
}

fn info(threads: Option<usize>) {
    for (k, v) in crate::vendor::identity().lines() {
        println!("{k}={v}")
    }
    println!(
        "host.logical_cpus={}",
        std::thread::available_parallelism().map_or(0, NonZeroUsize::get)
    );
    if let Some(n) = threads {
        println!("threads.requested={n}")
    }
    let mut features = Vec::new();
    if cfg!(feature = "link-openblas") {
        features.push("link-openblas")
    }
    if cfg!(feature = "link-openblas-static") {
        features.push("link-openblas-static")
    }
    if cfg!(feature = "link-mkl") {
        features.push("link-mkl")
    }
    println!("compiled.features={}", features.join(","));
}

const CSV_HEADER: &str =
    "regime,family,dtype,m,n,batch,row,threads,total_ms,per_item_us,status,note,\
samples_ms,pool_threads,vendor_threads";

fn write_csv(path: &str, records: &[rows::Record]) -> Result<(), String> {
    let mut text = String::from(CSV_HEADER);
    text.push('\n');
    for r in records {
        text.push_str(&r.csv());
        text.push('\n')
    }
    std::fs::write(Path::new(path), text).map_err(|e| e.to_string())
}

/// Keep the rows this run declares and put them in the rotated order the campaign expects.
///
/// The evidence check compares each case's row order with the rotation recorded for that process
/// start, so the harness has to emit the declared order rather than whatever order it measured in.
fn arrange(records: &mut Vec<rows::Record>, rows: &[String], rotate_rows: usize) {
    if rows.is_empty() {
        return;
    }
    let rotation = rotate_rows % rows.len();
    let wanted: Vec<&String> = rows[rotation..]
        .iter()
        .chain(rows[..rotation].iter())
        .collect();
    records.retain(|r| rows.contains(&r.row));
    records.sort_by_key(|r| {
        wanted
            .iter()
            .position(|w| **w == r.row)
            .unwrap_or(usize::MAX)
    });
}

/// What every census row of a process shares: the regime label, the environment, the declared rows
/// and the rotation this process start uses.
struct Census<'a> {
    regime: &'a str,
    env: &'a Env,
    rows: &'a [String],
    rotate_rows: usize,
}

/// The verification census for one case: the same cells the timing run will emit, with their status.
fn census(ctx: &Census<'_>, family: &str, case: Case, dtype: &str, ok: bool) -> Vec<rows::Record> {
    let regime = ctx.regime;
    let env = ctx.env;
    let rows = ctx.rows;
    let rotate_rows = ctx.rotate_rows;
    let threads = env.threads();
    let mut records: Vec<rows::Record> = rows::plan(family, case, env)
        .into_iter()
        .map(|(row, applies, note)| rows::Record {
            regime: regime.into(),
            family: family.into(),
            dtype: dtype.into(),
            m: case.m,
            n: case.n,
            batch: case.batch,
            row,
            threads,
            samples_ms: Vec::new(),
            total_ms: 0.,
            per_item_us: 0.,
            status: if applies && ok {
                "ok".into()
            } else if applies {
                "failed".into()
            } else {
                "skipped".into()
            },
            note,
            pool_threads: threads,
            vendor_threads: threads,
        })
        .collect();
    arrange(&mut records, rows, rotate_rows);
    records
}

/// Execute the command line and return a process exit code.
pub fn main() -> ExitCode {
    let argv: Vec<String> = std::env::args().collect();
    let a = match parse(&argv) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("{e}\n{USAGE}");
            return ExitCode::from(2);
        }
    };
    // Before the identity is read, and before any numerical call: MKL's interface layer is an ABI
    // the environment can otherwise choose for it.
    if let Err(e) = crate::vendor::prepare() {
        eprintln!("{e}");
        return ExitCode::from(2);
    }
    if a.command == "info" {
        info(a.threads);
        return ExitCode::SUCCESS;
    }
    let threads = a.threads.unwrap();
    if let Err(e) = crate::vendor::set_threads(threads) {
        eprintln!("{e}");
        return ExitCode::from(2);
    }
    let env = Env::with_threads(NonZeroUsize::new(threads).unwrap());
    if a.command == "verify" && !crate::vendor::LINKED {
        eprintln!(
            "{} skipped: vendor feature is not compiled in",
            crate::vendor::LAPACK_ROW
        );
    }
    let cases: Vec<Case> = a
        .shapes
        .iter()
        .flat_map(|&(m, n)| a.batches.iter().map(move |&batch| Case { m, n, batch }))
        .collect();
    let mut records = Vec::new();
    let mut census_records = Vec::new();
    let census_ctx = Census {
        regime: &a.regime,
        env: &env,
        rows: &a.rows,
        rotate_rows: a.rotate_rows,
    };
    let mut ok = true;
    for dtype in a.dtypes {
        for family in &a.families {
            for &case in &cases {
                eprintln!("{} {} {}", dtype.label(), family, case);
                match dtype {
                    Dtype::F64 => {
                        if a.command == "verify" {
                            let clean = verify::verify::<f64>(
                                std::slice::from_ref(family),
                                std::slice::from_ref(&case),
                                &env,
                            );
                            census_records.extend(census(&census_ctx, family, case, "f64", clean));
                            ok &= clean
                        } else {
                            let mut measured = rows::measure_case::<f64>(
                                family,
                                case,
                                &env,
                                a.reps,
                                a.prime_ms,
                                &a.regime,
                                a.statistic,
                            );
                            arrange(&mut measured, &a.rows, a.rotate_rows);
                            records.extend(measured);
                        }
                    }
                    Dtype::C64 => {
                        if a.command == "verify" {
                            let clean = verify::verify::<num_complex::Complex64>(
                                std::slice::from_ref(family),
                                std::slice::from_ref(&case),
                                &env,
                            );
                            census_records.extend(census(&census_ctx, family, case, "c64", clean));
                            ok &= clean
                        } else {
                            let mut measured = rows::measure_case::<num_complex::Complex64>(
                                family,
                                case,
                                &env,
                                a.reps,
                                a.prime_ms,
                                &a.regime,
                                a.statistic,
                            );
                            arrange(&mut measured, &a.rows, a.rotate_rows);
                            records.extend(measured);
                        }
                    }
                }
            }
        }
    }
    if a.command == "verify" {
        // The campaign's evidence check reads this census: one row per cell the run will emit,
        // including the declared skips, with the same rotation the run uses.
        if let Some(path) = a.csv {
            if let Err(e) = write_csv(&path, &census_records) {
                eprintln!("{e}");
                return ExitCode::from(2);
            }
        }
        if ok {
            println!("all comparisons within tolerance");
            return ExitCode::SUCCESS;
        } else {
            return ExitCode::from(1);
        }
    }
    if let Some(path) = a.csv {
        if let Err(e) = write_csv(&path, &records) {
            eprintln!("{e}");
            return ExitCode::from(2);
        }
    } else {
        for r in records {
            println!(
                "{} {} {} {} {:.3} ms {}",
                r.status,
                r.family,
                r.dtype,
                r.row,
                r.total_ms,
                note_or_empty(&r.note)
            );
        }
    }
    ExitCode::SUCCESS
}
fn note_or_empty(s: &str) -> &str {
    if s.is_empty() {
        ""
    } else {
        s
    }
}
