import argparse
import logging
import jax
import sys

# logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

def test_pmap_mpi():
    jax.distributed.initialize()

    global_device_count = jax.device_count()
    local_device_count = jax.local_device_count()
    print(f"device_count = {global_device_count}")
    print(f"local_device_count = {local_device_count}")

    xs = jax.numpy.ones(jax.local_device_count())
    sum = jax.pmap(lambda x: jax.lax.psum(x, 'i'), axis_name='i')(xs)

    print(sum)

test_pmap_mpi()
