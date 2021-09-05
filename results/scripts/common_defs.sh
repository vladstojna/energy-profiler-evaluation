work_types=("alternating" "rng" "sleep" "mkl" "openblas")

function echoerr
{
    printf "%s\n" "$*" >&2
}

default_work="all"

function usage_w_string
{
    local var=$(printf "$1%s" "${work_types[@]}")
    echo ${var:1}
}

function is_candidate
{
    [[ "$1" == "$2" ]] || [[ "$1" == "$default_work" ]]
}

function is_valid_work
{
    for w in "${work_types[@]}"; do
        if [[ "$1" == "$w" ]]; then
            true
            return
        fi
    done
    false
}

samples_dir=$(cd $(dirname "$0")/../.. && pwd)
threads=$(lscpu -b -p=CORE | grep -v '^#' | wc -l)
cores=$(lscpu -b -p=CORE | grep -v '^#' | sort -u | wc -l)
