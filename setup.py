from glob import glob
import os

from setuptools import find_packages, setup

package_name = 'marine_project'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'),
            glob('launch/*.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='root',
    maintainer_email='pedemari.rr@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': ["dynamics_node = marine_project.dynamics_node:main", 
        "PID = marine_project.PID:main",
        "SMC = marine_project.SMC:main",
        "currents = marine_project.currents:main"
        ],
    },
)
