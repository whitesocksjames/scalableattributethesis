#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
    echo "Usage: $0 HPC_SOURCE N30_DESTINATION" >&2
    exit 2
fi

source_path=$1
destination_path=$2

hpc_root=/home/woody/iwnt/iwnt193h
n30_root=/data/run01/scz0ade/Tanzeyu
credential_dir=$n30_root/.ssh_transfer
identity_file=$credential_dir/id_rsa_fau_hpc
known_hosts_file=$credential_dir/known_hosts

case "$source_path" in
    "$hpc_root" | "$hpc_root"/*) ;;
    *)
        echo "Refusing source outside $hpc_root: $source_path" >&2
        exit 2
        ;;
esac

case "$destination_path" in
    "$n30_root" | "$n30_root"/*) ;;
    *)
        echo "Refusing destination outside $n30_root: $destination_path" >&2
        exit 2
        ;;
esac

if [[ ! -f "$identity_file" ]]; then
    echo "Missing FAU identity file: $identity_file" >&2
    exit 1
fi

install -d -m 700 "$credential_dir"
chmod 600 "$identity_file"
mkdir -p "$destination_path"

proxy_command="ssh -i $identity_file -o BatchMode=yes -o ConnectTimeout=20 -o UserKnownHostsFile=$known_hosts_file -W %h:%p iwnt193h@csnhr.nhr.fau.de"
remote_shell="ssh -i $identity_file -o BatchMode=yes -o ConnectTimeout=20 -o UserKnownHostsFile=$known_hosts_file -o ProxyCommand=\"$proxy_command\""

rsync \
    --archive \
    --human-readable \
    --itemize-changes \
    --partial \
    --append-verify \
    --info=progress2 \
    -e "$remote_shell" \
    "iwnt193h@tinyx.nhr.fau.de:$source_path" \
    "$destination_path"
