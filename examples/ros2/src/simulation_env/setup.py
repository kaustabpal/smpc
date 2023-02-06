from setuptools import setup

package_name = 'simulation_env'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='adi99',
    maintainer_email='meduri99aditya@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'ngsim_env = simulation_env.ngsim_env:main',
            'combined_sim_new = simulation_env.combined_sim_new:main',
            'ngsim_acado = simulation_env.ngsim_acado:main',
            'combined_sim_acado = simulation_env.combined_sim_acado:main',
            'behaviour_multi_goal = simulation_env.behaviour_multi_goal:main',
        ],
    },
)
