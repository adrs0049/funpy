#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author: Andreas Buttenschoen
import re

def all_keys(dictlist):
    return set().union(*dictlist)

class Namespace(dict):
    def __init__(self):
        pass

    def writeHDF5(self, fh, group_name='ns'):
        grp = fh.create_group(group_name)
        for k, v in self.items():
            grp.attrs[k] = v

    def readHDF5(self, fh, group_name='ns', *args, **kwargs):
        for k, v in fh[group_name].attrs.items():
            self[k] = v

    def __str__(self):
        rstr = 'Namespace:'
        for k, v in self.items():
            rstr += '{} '.format(str(v))
        return rstr

    def __repr__(self):
        return self.__str__()

def parse_parameter(config_line):
    match = re.search(
        r"""^          # Anchor to start of line
        (\s*)          # $1: Zero or more leading ws chars
        (?:            # Begin group for optional var=value.
          (\S+)        # $2: Variable name. One or more non-spaces.
          (\s*=\s*)    # $3: Assignment operator, optional ws
          (            # $4: Everything up to comment or EOL.
            [^#\\]*    # Unrolling the loop 1st normal*.
            (?:        # Begin (special normal*)* construct.
              \\.      # special is backslash-anything.
              [^#\\]*  # More normal*.
            )*         # End (special normal*)* construct.
          )            # End $4: Value.
        )?             # End group for optional var=value.
        ((?:\#.*)?)    # $5: Optional comment.
        $              # Anchor to end of line""",
        config_line, re.MULTILINE | re.VERBOSE)

    if match is None:
        return None

    return match.groups()


""" Reads parameter par_name from the config file """
def parse_config(config_file, par_name):
    with open(config_file, 'r') as f:
        content = f.readlines()
        content = [x.strip() for x in content]

    for line in content:
        groups = parse_parameter(line)

        if groups is None:
            continue

        pn        = groups[1]
        par_value = groups[3]

        if par_name == pn:
            return float(par_value)

    return None

def get_required_symbols(src, default_value=1.):
    # find required variables
    required_vars = []

    # local copy of source
    rhs = src

    while True:
        try:
            # create a local namespace
            ns = Namespace()

            exec(rhs, ns)
        except NameError as e:
            # parse the exception
            m   = re.search('\'(.+?)\'', e.args[0])
            var = m.group(1)

            # we found something missing!
            required_vars.append(var)

            # prepend some value
            rhs = "{0} = {1}".format(var, default_value) + "; " + rhs

        except Exception as e:
            raise e
        else:
            break

    return set(required_vars)
