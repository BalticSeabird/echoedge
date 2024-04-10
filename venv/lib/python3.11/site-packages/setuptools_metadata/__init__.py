import os
import os.path
import subprocess
import sys

import distutils.core

# ./setup.py metadata --key=name

# setup(
#   ...
#   custom_metadata={
#       x_thing_1: ['a', 'b', 'c'],
#       x_some_other_thing: 'bla',
#       x_whatever: 123
#   })

class metadata(distutils.core.Command):
    description = 'query setup.py metadata'

    user_options = [
        ('key=', 'k', 'setup() key name to query')
    ]

    def initialize_options(self):
        self.key = None

    def finalize_options(self):
        if self.key is None:
            raise Exception('Option --key not given')

    def run(self):
        result = None

        if self.key in self.distribution.metadata._METHOD_BASENAMES:
            result = getattr(self.distribution.metadata, self.key)

        elif hasattr(self.distribution, self.key):
            result = getattr(self.distribution, self.key)

        elif self.key in self.distribution.custom_metadata:
            result = self.distribution.custom_metadata[self.key]

        if isinstance(result, list):
            sys.stdout.write('\n'.join(result))
        else:
            sys.stdout.write(str(result))

        if sys.stdout.isatty():
            sys.stdout.write('\n')

def validate_dict(dist, attr, value):
    if not isinstance(value, dict):
        raise Exception('custom_metadata should be a dict')
    for key in value:
        if not isinstance(key, str):
            raise Exception('custom_metadata key {0!r} should be a str', key)
        if not key.startswith('x_'):
            raise Exception('custom_metadata key {0!r} should start with x_', key)

