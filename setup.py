from distutils.core import setup


with open('README.md', 'rb') as f:
    readme = f.read().decode('utf8')

setup(name='autograd',
      version='1.0.0',
      author='Mario Sasko',
      author_email='mariosasko777@gmail.com',
      url='https://github.com/mariosasko/autograd',
      packages=['autograd'], 
      description='A reverse-mode automatic differentiation package',
      long_description=readme,
      license='MIT',
     )