#!/usr/bin/env bash
#
# Install the built tlinalg shared library, its header, a pkg-config file and the
# licences into a prefix.
#
#   crates/tlinalg-capi/install.sh --prefix=/opt/tlinalg
#
# The layout is produced by a manual site install, by CI and by a BinaryBuilder
# recipe alike, and this script is the single definition of it. It does not build
# anything and it does not rewrite binaries: the SONAME and the Mach-O install
# name are set at link time, because rustc emits neither for a `cdylib`. This
# script checks for them and warns rather than papering over their absence.

set -euo pipefail

crate_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${crate_dir}/../.." && pwd)"

prefix=""
libdir=""
includedir=""
artifacts="${repo_root}/target/release"
version=""
licenses=1

usage() {
    cat >&2 <<EOF
usage: install.sh --prefix=DIR [options]

  --prefix=DIR      installation prefix (required)
  --libdir=DIR      library directory (default: PREFIX/lib, or PREFIX/bin for a DLL)
  --includedir=DIR  header directory (default: PREFIX/include)
  --artifacts=DIR   where cargo left the library (default: target/release)
  --version=X.Y.Z   version for the pkg-config file (default: read from Cargo.toml)
  --no-licenses     skip the licences
EOF
    exit 2
}

for arg in "$@"; do
    case "${arg}" in
        --prefix=*)     prefix="${arg#*=}" ;;
        --libdir=*)     libdir="${arg#*=}" ;;
        --includedir=*) includedir="${arg#*=}" ;;
        --artifacts=*)  artifacts="${arg#*=}" ;;
        --version=*)    version="${arg#*=}" ;;
        --no-licenses)  licenses=0 ;;
        -h|--help)      usage ;;
        *)              echo "install.sh: unknown argument '${arg}'" >&2; usage ;;
    esac
done

[[ -n "${prefix}" ]] || usage
[[ -d "${artifacts}" ]] || {
    echo "install.sh: no such artifact directory: ${artifacts}" >&2
    echo "            run 'cargo build --release -p tlinalg-capi' first" >&2
    exit 1
}

# The version is the workspace's, which this crate inherits, so it is read from
# the workspace manifest: the single source of truth the header macro and
# tlinalg_abi_version() are both checked against.
if [[ -z "${version}" ]]; then
    version="$(sed -n 's/^version = "\(.*\)"$/\1/p' "${repo_root}/Cargo.toml" | head -n1)"
    [[ -n "${version}" ]] || { echo "install.sh: cannot read version from the workspace Cargo.toml" >&2; exit 1; }
fi

# Find the library by its filename rather than by guessing the host: this script
# runs under cross-compilation, where the host says nothing about the artifact.
lib=""
for candidate in \
    "${artifacts}/libtlinalg_capi.so" \
    "${artifacts}/libtlinalg_capi.dylib" \
    "${artifacts}/tlinalg_capi.dll"
do
    if [[ -f "${candidate}" ]]; then lib="${candidate}"; break; fi
done
[[ -n "${lib}" ]] || {
    echo "install.sh: no shared library in ${artifacts}" >&2
    echo "            expected libtlinalg_capi.{so,dylib} or tlinalg_capi.dll" >&2
    echo "            on musl this means the cdylib was dropped: build with" >&2
    echo "            RUSTFLAGS='-C target-feature=-crt-static'" >&2
    exit 1
}
libname="$(basename "${lib}")"

if [[ -z "${libdir}" ]]; then
    case "${libname}" in
        *.dll) libdir="${prefix}/bin" ;;
        *)     libdir="${prefix}/lib" ;;
    esac
fi
[[ -n "${includedir}" ]] || includedir="${prefix}/include"

install -d "${libdir}" "${includedir}" "${prefix}/lib/pkgconfig"
[[ ${licenses} -eq 1 ]] && install -d "${prefix}/share/licenses/tlinalg"

install -m 0755 "${lib}" "${libdir}/${libname}"
echo "installed ${libdir}/${libname}"

if [[ "${libname}" == *.dll && -f "${artifacts}/libtlinalg_capi.dll.a" ]]; then
    install -m 0644 "${artifacts}/libtlinalg_capi.dll.a" \
                    "${prefix}/lib/libtlinalg_capi.dll.a"
    echo "installed ${prefix}/lib/libtlinalg_capi.dll.a"
fi

(cd "${crate_dir}/include" && find . -type f -name '*.h') | while read -r header; do
    install -D -m 0644 "${crate_dir}/include/${header}" "${includedir}/${header#./}"
done
echo "installed headers under ${includedir}"

sed -e "s|@PREFIX@|${prefix}|g" \
    -e "s|@LIBDIR@|${libdir}|g" \
    -e "s|@INCLUDEDIR@|${includedir}|g" \
    -e "s|@VERSION@|${version}|g" \
    "${crate_dir}/tlinalg_capi.pc.in" \
    > "${prefix}/lib/pkgconfig/tlinalg_capi.pc"
echo "installed ${prefix}/lib/pkgconfig/tlinalg_capi.pc"

if [[ ${licenses} -eq 1 ]]; then
    for licence in LICENSE-MIT LICENSE-APACHE; do
        install -m 0644 "${repo_root}/${licence}" \
                        "${prefix}/share/licenses/tlinalg/${licence}"
    done
    echo "installed ${prefix}/share/licenses/tlinalg/LICENSE-{MIT,APACHE}"
fi

case "${libname}" in
    *.so)
        if command -v readelf > /dev/null; then
            soname="$(readelf -d "${libdir}/${libname}" \
                      | sed -n 's/.*SONAME.*\[\(.*\)\]/\1/p')"
            if [[ -z "${soname}" ]]; then
                echo "install.sh: warning: no DT_SONAME." >&2
                echo "            rustc does not emit one for a cdylib; rebuild with" >&2
                echo "            RUSTFLAGS='-C link-arg=-Wl,-soname,${libname}'" >&2
            else
                echo "soname: ${soname}"
            fi
        fi
        ;;
    *.dylib)
        if command -v otool > /dev/null; then
            id="$(otool -D "${libdir}/${libname}" | tail -n1 | tr -d '[:space:]')"
            case "${id}" in
                @rpath/*|"${libname}") echo "install name: ${id}" ;;
                *) echo "install.sh: warning: install name is '${id}', not relocatable." >&2
                   echo "            rebuild with RUSTFLAGS='-C link-arg=-Wl,-install_name,@rpath/${libname}'" >&2 ;;
            esac
        fi
        ;;
esac
