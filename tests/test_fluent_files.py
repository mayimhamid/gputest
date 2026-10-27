"""FlowTrace's Fluent readers and writers against files written by Ansys Fluent (opt-in).

The reference files (PyFluent mixing-elbow example, ansys/example-data, MIT licence) are not part
of the repository. Fetch them once with ``python benchmarks/fluent_files_check.py`` (into
~/.flowtrace/fluent_examples, checksums verified) or point FLOWTRACE_FLUENT_EXAMPLES at a folder
that holds them; otherwise this test is skipped.
"""

import pytest

from flowtrace.validation.fluent_files import check, find, verdicts

pytest.importorskip("h5py")
PATHS = find()


@pytest.mark.skipif(PATHS is None, reason="Fluent reference files not present (see the module docstring)")
def test_against_files_written_by_fluent(tmp_path):
    result = check(PATHS, workdir=str(tmp_path))
    failed = [k for k, ok in verdicts(result).items() if not ok]
    assert not failed, (failed, result)
