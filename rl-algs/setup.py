from setuptools import setup, find_packages

setup(name='rl-algs',
      py_modules=['rl_algs'],
      install_requires=[
          'scipy',
          'tqdm',
          'joblib',
      ]
      description="OpenAI baselines: high quality implementations of reinforcement learning algorithms",
      author="OpenAI",
)
