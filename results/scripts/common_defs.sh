function echoerr
{
    printf "%s\n" "$*" >&2
}

default_work="all"

function is_candidate
{
    [[ $1 == $2 ]] || [[ $1 == $default_work ]]
}

function is_valid_work
{
    if [[ "$1" == @(alternating|rng|sleep|mkl|openblas) ]]; then
        true
    else
        echoerr "Invalid work type $1"
        false
    fi
}

samples_dir=$(cd $(dirname "$0")/../.. && pwd)
threads=$(lscpu -b -p=CORE | grep -v '^#' | wc -l)
cores=$(lscpu -b -p=CORE | grep -v '^#' | sort -u | wc -l)
