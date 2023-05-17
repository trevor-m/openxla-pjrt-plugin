import argparse
import logging
import jax
import sys

# logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

def test_distributed_pmap(coordinator_address_, process_id_, num_processes_,
                          local_device_ids_):
    jax.distributed.initialize(coordinator_address=coordinator_address_,
                               num_processes=num_processes_,
                               process_id=process_id_,
                               local_device_ids=local_device_ids_)
    print(f"num_processes = {num_processes_}")
    print(f"process_id = {process_id_}")
    print(f"device_count = {jax.device_count()}")
    print(f"local_device_count = {jax.local_device_count()}")

    xs = jax.numpy.ones(jax.local_device_count())
    sum = jax.pmap(lambda x: jax.lax.psum(x, 'i'), axis_name='i')(xs)
    if (process_id_ == 0):
        print(sum)

parser = argparse.ArgumentParser()
parser.add_argument("coordinator_address", help="coordinator address")
parser.add_argument("procid", type=int, help="process ID")
parser.add_argument("nprocs", type=int, help="number of processes")
parser.add_argument("local_device_id", type=int, help="local device ID")

args = parser.parse_args()
    
test_distributed_pmap(args.coordinator_address,
                      args.procid,
                      args.nprocs,
                      [args.local_device_id])
