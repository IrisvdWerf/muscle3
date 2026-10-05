from pathlib import Path

import pytest
from ymmsl.v0_2 import (
    Conduit,
    Configuration,
    ExecutionModel,
    Identifier,
    Model,
    Operator,
    Port,
    Ports,
    Program,
    Reference,
    Timeline,
    resolve_timelines,
)

from libmuscle.pytest.muscle_tester import MuscleTester


@pytest.fixture
def tmp_run_dir(tmp_path: Path) -> Path:
    run_dir = tmp_path / "run_dir"
    run_dir.mkdir()
    return run_dir


@pytest.fixture
def program_config() -> Configuration:
    """Configuration whose implementation-under-test is a Program."""
    prog = Program(
        name="micro_model_program",
        ports=Ports(f_init=["init_in"], o_f=["final_out"]),
        execution_model=ExecutionModel.MANUAL,
    )
    return Configuration(programs=[prog])


@pytest.fixture
def model_config() -> Configuration:
    """Configuration whose implementation-under-test is a Model."""
    sub_model = Model(
        name="macro_model",
        ports=Ports(o_i=["state_out"], s=["update_in"]),
    )
    return Configuration(models=[sub_model])


@pytest.fixture
def meso_program_config() -> Configuration:
    """Configuration whose implementation-under-test is a Program with all
    operators, with its O_I and S ports on two different timelines."""
    prog = Program(
        name="meso_model_program",
        ports=Ports(
            [
                Port(Identifier("init_in"), Operator.F_INIT),
                Port(Identifier("state_out"), Operator.O_I, Timeline("sub1")),
                Port(Identifier("update_in"), Operator.S, Timeline("sub2")),
                Port(Identifier("final_out"), Operator.O_F),
            ]
        ),
        execution_model=ExecutionModel.MANUAL,
    )
    return Configuration(programs=[prog])


@pytest.fixture
def meso_model_config() -> Configuration:
    """Configuration whose implementation-under-test is a Model with all operators,
    with its O_I and S ports on two different timelines."""
    sub_model = Model(
        name="meso_model",
        ports=Ports(
            [
                Port(Identifier("init_in"), Operator.F_INIT),
                Port(Identifier("state_out"), Operator.O_I, Timeline("sub1")),
                Port(Identifier("update_in"), Operator.S, Timeline("sub2")),
                Port(Identifier("final_out"), Operator.O_F),
            ]
        ),
    )
    return Configuration(models=[sub_model])


def test_add_tester_model_to_config(
    tmp_run_dir: Path, meso_model_config: Configuration
) -> None:
    tester = MuscleTester(tmp_run_dir)
    result = tester._add_tester_component(meso_model_config, "meso_model")

    tester_model = result.models[Reference("muscle3_test_model")]
    assert set(tester_model.components) == {
        Reference("muscle3_implementation_tester_parent"),
        Reference("muscle3_implementation_tester_sibling"),
        Reference("meso_model"),
    }


def test_add_tester_program_to_config(
    tmp_run_dir: Path, meso_program_config: Configuration
) -> None:
    tester = MuscleTester(tmp_run_dir)
    result = tester._add_tester_component(meso_program_config, "meso_model_program")

    for name in (
        "muscle3_implementation_tester_parent",
        "muscle3_implementation_tester_sibling",
    ):
        tester_prog = result.programs[Reference(name)]
        assert tester_prog.execution_model == ExecutionModel.MANUAL


def test_add_test_ports_to_config(
    tmp_run_dir: Path, meso_program_config: Configuration
) -> None:
    tester = MuscleTester(tmp_run_dir)
    result = tester._add_tester_component(meso_program_config, "meso_model_program")
    tester_model = result.models[Reference("muscle3_test_model")]

    def ports(tester_name: str) -> dict[str, tuple[Operator, Timeline]]:
        component = tester_model.components[Reference(tester_name)]
        return {
            str(name): (port.operator, port.timeline)
            for name, port in component.ports.items()
        }

    # The parent mirrors the F_INIT and O_F ports, and nests the sibling
    assert ports("muscle3_implementation_tester_parent") == {
        "init_in": (Operator.O_I, Timeline([])),
        "final_out": (Operator.S, Timeline([])),
        "__sibling_settings_out__": (Operator.O_I, Timeline([])),
    }

    # The sibling mirrors the O_I and S ports, on the same timelines
    assert ports("muscle3_implementation_tester_sibling") == {
        "state_out": (Operator.S, Timeline("sub1")),
        "update_in": (Operator.O_I, Timeline("sub2")),
    }


def test_add_test_conduits_to_config(
    tmp_run_dir: Path, meso_program_config: Configuration
) -> None:
    tester = MuscleTester(tmp_run_dir)
    result = tester._add_tester_component(meso_program_config, "meso_model_program")
    tester_model = result.models[Reference("muscle3_test_model")]

    parent = "muscle3_implementation_tester_parent"
    sibling = "muscle3_implementation_tester_sibling"
    expected = [
        # The parent sends on F_INIT and receives from O_F
        Conduit(f"{parent}.init_in", "meso_model_program.init_in"),
        Conduit("meso_model_program.final_out", f"{parent}.final_out"),
        # The sibling receives from O_I and sends on S
        Conduit("meso_model_program.state_out", f"{sibling}.state_out"),
        Conduit(f"{sibling}.update_in", "meso_model_program.update_in"),
        # The parent nests the sibling in its timeline
        Conduit(f"{parent}.__sibling_settings_out__", f"{sibling}.muscle_settings_in"),
    ]
    assert len(tester_model.conduits) == len(expected)
    for conduit in expected:
        assert conduit in tester_model.conduits

    # Via F_INIT, the parent nests the implementation next to the sibling.
    resolve_timelines(tester_model)
    assert {
        str(name): component.timeline
        for name, component in tester_model.components.items()
    } == {
        parent: Timeline(parent),
        "meso_model_program": Timeline(f"{parent}:meso_model_program"),
        sibling: Timeline(f"{parent}:{sibling}"),
    }


def test_original_config_unchanged(
    tmp_run_dir: Path, program_config: Configuration
) -> None:
    """add_tester_component should not remove the original model/program."""
    tester = MuscleTester(tmp_run_dir)
    result = tester._add_tester_component(program_config, "micro_model_program")
    assert Reference("micro_model_program") in result.programs


def test_error_for_unknown_implementation(tmp_run_dir: Path) -> None:
    config = Configuration(models=[], programs=[])
    tester = MuscleTester(tmp_run_dir)
    with pytest.raises(ValueError, match="No implementation 'nonexistent'"):
        tester._add_tester_component(config, "nonexistent")


def test_add_settings_conduit_without_f_init(
    tmp_run_dir: Path, model_config: Configuration
) -> None:
    tester = MuscleTester(tmp_run_dir)
    result = tester._add_tester_component(model_config, "macro_model")
    tester_model = result.models[Reference("muscle3_test_model")]

    # Without F_INIT ports, the parent nests the implementation via muscle_settings_in
    parent = "muscle3_implementation_tester_parent"
    assert (
        Conduit(f"{parent}.__settings_in__", "macro_model.muscle_settings_in")
        in tester_model.conduits
    )

    # This raises if the timelines are not consistent
    resolve_timelines(tester_model)
    implementation = tester_model.components[Reference("macro_model")]
    assert implementation.timeline == Timeline(f"{parent}:macro_model")


def test_add_matching_timelines(
    tmp_run_dir: Path, meso_program_config: Configuration
) -> None:
    tester = MuscleTester(tmp_run_dir)
    result = tester._add_tester_component(meso_program_config, "meso_model_program")
    tester_model = result.models[Reference("muscle3_test_model")]

    # Each timeline of the implementation matches the same one of the sibling
    parent = "muscle3_implementation_tester_parent"
    sibling = "muscle3_implementation_tester_sibling"
    assert tester_model.matching_timelines is not None
    assert [match.matches for match in tester_model.matching_timelines] == [
        {
            Timeline(f"{parent}:meso_model_program.sub1"),
            Timeline(f"{parent}:{sibling}.sub1"),
        },
        {
            Timeline(f"{parent}:meso_model_program.sub2"),
            Timeline(f"{parent}:{sibling}.sub2"),
        },
    ]


def test_no_sibling_without_o_i_and_s_ports(
    tmp_run_dir: Path, program_config: Configuration
) -> None:
    tester = MuscleTester(tmp_run_dir)
    result = tester._add_tester_component(program_config, "micro_model_program")

    tester_model = result.models[Reference("muscle3_test_model")]
    assert set(tester_model.components) == {
        Reference("muscle3_implementation_tester_parent"),
        Reference("micro_model_program"),
    }
    assert Reference("muscle3_implementation_tester_sibling") not in result.programs
