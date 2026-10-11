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

const USAGE: &str = "usage: tlbench info | verify --threads N --n LIST [--batch LIST] [--dtype LIST] [--family LIST] | run --threads N --n LIST [--batch LIST] [--dtype LIST] [--family LIST] [--regime LABEL] [--reps M] [--prime-ms MS] [--csv PATH]";

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

fn write_csv(path: &str, records: &[rows::Record]) -> Result<(), String> {
    let mut text = String::from(
        "regime,family,dtype,m,n,batch,row,threads,total_ms,per_item_us,status,note\n",
    );
    for r in records {
        text.push_str(&r.csv());
        text.push('\n')
    }
    std::fs::write(Path::new(path), text).map_err(|e| e.to_string())
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
    let mut ok = true;
    for dtype in a.dtypes {
        for family in &a.families {
            for &case in &cases {
                eprintln!("{} {} {}", dtype.label(), family, case);
                match dtype {
                    Dtype::F64 => {
                        if a.command == "verify" {
                            ok &= verify::verify::<f64>(
                                std::slice::from_ref(family),
                                std::slice::from_ref(&case),
                                &env,
                            )
                        } else {
                            records.extend(rows::measure_case::<f64>(
                                family, case, &env, a.reps, a.prime_ms, &a.regime,
                            ));
                        }
                    }
                    Dtype::C64 => {
                        if a.command == "verify" {
                            ok &= verify::verify::<num_complex::Complex64>(
                                std::slice::from_ref(family),
                                std::slice::from_ref(&case),
                                &env,
                            )
                        } else {
                            records.extend(rows::measure_case::<num_complex::Complex64>(
                                family, case, &env, a.reps, a.prime_ms, &a.regime,
                            ));
                        }
                    }
                }
            }
        }
    }
    if a.command == "verify" {
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
