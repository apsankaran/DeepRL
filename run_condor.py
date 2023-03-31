from __future__ import print_function
from __future__ import division

import sys
import os
import argparse
import subprocess
import random
import time
import pdb
import itertools

parser = argparse.ArgumentParser()

# saving
parser.add_argument('result_directory', default = None, help='Directory to write results to.')

# common setup
parser.add_argument('--env_name', type = str, required = True)
parser.add_argument('--num_timesteps', default = 4000000, type = int, required = True)

# variables
parser.add_argument('--num_trials', default = 10, type=int, help='The number of trials to launch.')
parser.add_argument('--condor', default = False, action='store_true', help='run experiments on condor')

FLAGS = parser.parse_args()

ct = 0
EXECUTABLE = 'job.sh'

def get_cmd(gae_lambda,
            seed,
            outfile,
            condor = False):
  
    arguments = '--outfile %s --seed %d --gae_lambda %f' % (outfile, seed, gae_lambda)
    arguments += ' --env_name %s' % FLAGS.env_name
    arguments += ' --num_timesteps %d' % FLAGS.num_timesteps
   
    if FLAGS.condor:
        cmd = '%s' % (arguments)
    else:
        EXECUTABLE = 'run_single_cont.py'
        cmd = 'python3 %s %s' % (EXECUTABLE, arguments)
    return cmd

def run_trial(gae_lambda, seed,
            outfile,
            condor = False):

    cmd = get_cmd(gae_lambda, seed,
                outfile)
    if condor:
        submitFile = 'universe = vanilla\n'
        submitFile += 'executable = ' + EXECUTABLE + "\n"
        submitFile += 'arguments = ' + cmd + '\n'
        submitFile += 'error = %s.err\n' % outfile
        #submitFile += 'log = %s.log\n' % outfile
        submitFile += 'log = /dev/null\n'
        submitFile += 'output = /dev/null\n'
        #submitFile += 'output = %s.out\n' % outfile
        submitFile += 'should_transfer_files = YES\n'
        submitFile += 'when_to_transfer_output = ON_EXIT\n'

        setup_files = 'http://proxy.chtc.wisc.edu/SQUID/apsankaran/research1.tar.gz'
        # common_main_files = 'run_single_cont.py, continuous_density_ratio.py, estimators.py, policies.py, utils.py, run_single_learn_phi.py, learn_phi.py'
        common_main_files = 'StableBaselines3.py, CustomPPOModel2.py, CustomOnPolicyAlgorithm.py, CustomCallback.py'
        # domains = 'infinite_walker.py, walker, infinite_pusher.py, pusher, infinite_antumaze.py, antumaze, infinite_reacher.py, reacher, a2c_ppo_acktr'

        # submitFile += 'transfer_input_files = {}, {}, {}\n'.format(setup_files, common_main_files, domains)
        submitFile += 'transfer_input_files = {}, {}\n'.format(setup_files, common_main_files)
        submitFile += 'requirements = (has_avx == True)\n'
        submitFile += 'request_cpus = 1\n'
        submitFile += 'request_memory = 5GB\n'
        submitFile += 'request_disk = 7GB\n'
        submitFile += 'queue'

        proc = subprocess.Popen('condor_submit', stdin=subprocess.PIPE)
        proc.stdin.write(submitFile.encode())
        proc.stdin.close()
        time.sleep(0.2)
    else:
        # TODO
        # pdb.set_trace()
        #subprocess.run('"conda init bash; conda activate research; {}"'.format(cmd), shell=True)
        #cmd = 'bash -c "source activate root"' 
        subprocess.Popen(('conda run -n research ' + cmd).split())

def _launch_trial(gae_lambdas, seeds):

    global ct
    for gae_lambda in gae_lambdas:
        for seed in seeds:
            outfile = "env_{}_gae_lambda_{}_seed_{}".format(FLAGS.env_name, gae_lambda, seed)
            if os.path.exists(outfile):
                continue
            run_trial(gae_lambda, seed, outfile, condor=FLAGS.condor)
            ct += 1
            print('submitted job number: %d' % ct)

def main():  # noqa

    if FLAGS.result_directory is None:
        print('Need to provide result directory')
        sys.exit()
    directory = FLAGS.result_directory + '_' + FLAGS.env_name + '_experiments'

    if not os.path.exists(directory):
        os.makedirs(directory)

    seeds = [random.randint(0, 1e6) for _ in range(FLAGS.num_trials)]
    gae_lambdas = [0, 0.5, 1]

    _launch_trial(gae_lambdas, seeds)

    print('%d experiments ran.' % ct)

if __name__ == "__main__":
    main()

