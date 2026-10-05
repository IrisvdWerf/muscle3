import multiprocessing as mp
import os
from collections.abc import Generator
from contextlib import ExitStack, contextmanager
from multiprocessing.connection import Connection
from pathlib import Path
from types import TracebackType
from unittest.mock import patch

import ymmsl.v0_2
from ymmsl.v0_2 import (
    Component,
    Conduit,
    Configuration,
    ExecutionModel,
    Identifier,
    Implementation,
    MatchingTimelines,
    Model,
    Operator,
    Port,
    Ports,
    Program,
    Reference,
    Timeline,
    resolve_timelines,
)

from libmuscle.manager.hammer import flatten
from libmuscle.manager.manager import Manager
from libmuscle.manager.run_dir import RunDir
from libmuscle.mcp.tcp_transport_client import RECONNECT_TIMEOUT
from libmuscle.mcp.tcp_transport_server import TcpTransportServer
from libmuscle.mmp_client import PEER_TIMEOUT
from libmuscle.pytest.implementation_tester import (
    PARENT_SETTINGS_PORT_NAME,
    PARENT_TESTER_NAME,
    SIBLING_SETTINGS_PORT_NAME,
    SIBLING_TESTER_NAME,
    ImplementationTester,
)
from libmuscle.receive_timeout_handler import ReceiveTimeoutHandler

TEST_MODEL_NAME = "muscle3_test_model"


def raise_error(*args: object) -> None:
    raise RuntimeError(args)


class MuscleTester:
    """Helper class to test an implementation.

    Note: You don't need to construct a MuscleTester directly; use the
    ``muscle3_tester`` pytest fixture instead.
    """

    def __init__(self, run_dir: Path) -> None:
        self.run_dir = run_dir
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.implementation_tester: ImplementationTester | None = None
        self._exitstack = ExitStack()

    def __enter__(self) -> "MuscleTester":
        """Allows usage in a with-statement"""
        return self

    def __exit__(
        self,
        typ: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        """Allows usage in a with-statement"""
        self.cleanup()

    def _add_tester_component(
        self, config: Configuration, implementation_name: str
    ) -> Configuration:
        """
        Add tester components connected to the implementation.
        - Finds the implementation (model or program) by name.
        - Adds tester components with MANUAL execution, ports and conduits.

        The ports of the implementation are connected to two tester components:
        - F_INIT and O_F ports are connected to the O_I and S ports of the
          'muscle3_implementation_tester_parent', which acts as the parent of the
          implementation.
        - O_I and S ports are connected to the S and O_I ports of the
          'muscle3_implementation_tester_sibling', like in an interact coupling,
          and their timelines are declared to match. The parent nests the sibling
          in its timeline through the sibling's muscle_settings_in, so that it is
          on the same level as the implementation.
        """

        implementation: Implementation
        if implementation_name in config.models:
            implementation = config.models[Reference(implementation_name)]
        elif implementation_name in config.programs:
            implementation = config.programs[Reference(implementation_name)]
        else:
            raise ValueError(
                f"No implementation '{implementation_name}' found in the yMMSL"
            )

        tester_model = Model(name=TEST_MODEL_NAME)

        parent_ports = self._add_parent_tester(
            tester_model, implementation_name, implementation
        )
        sibling_ports = self._add_sibling_tester(
            tester_model, implementation_name, implementation, parent_ports
        )

        programs = {PARENT_TESTER_NAME: parent_ports}
        if sibling_ports:
            programs[SIBLING_TESTER_NAME] = sibling_ports

        for name, ports in programs.items():
            tester_model.components[Reference(name)] = Component(
                name=name,
                ports=Ports(ports),
                description="Tester component for implementation testing",
                implementation=name,
                optional=False,
            )

        tester_model.components[Reference(implementation_name)] = Component(
            name=implementation_name,
            ports=implementation.ports,
            description="The tested implementation",
            implementation=implementation_name,
            optional=False,
        )

        for name, ports in programs.items():
            config.programs[Reference(name)] = Program(
                name=name,
                ports=Ports(ports),
                execution_model=ExecutionModel.MANUAL,
                description="Manual tester program for implementation testing",
            )

        config.models[Reference(TEST_MODEL_NAME)] = tester_model
        return config

    @staticmethod
    def _add_parent_tester(
        tester_model: Model, implementation_name: str, implementation: Implementation
    ) -> list[Port]:
        """Connect the F_INIT and O_F ports of the implementation to the parent.

        Args:
            tester_model: The test model to add conduits to.
            implementation_name: Name of the implementation.
            implementation: The implementation to test.

        Returns:
            The ports of the parent tester.
        """
        parent_ports = [
            MuscleTester._connect_mirrored_port(
                tester_model, PARENT_TESTER_NAME, implementation_name, port
            )
            for port in implementation.ports.values()
            if port.operator in (Operator.F_INIT, Operator.O_F)
        ]

        if not any(
            p.operator is Operator.F_INIT for p in implementation.ports.values()
        ):
            # We'll connect muscle_settings_in to make the timeline logic work
            tester_model.conduits.append(
                Conduit(
                    f"{PARENT_TESTER_NAME}.{PARENT_SETTINGS_PORT_NAME}",
                    f"{implementation_name}.muscle_settings_in",
                )
            )
            parent_ports.append(
                Port(Identifier(PARENT_SETTINGS_PORT_NAME), Operator.O_I)
            )

        return parent_ports

    @staticmethod
    def _add_sibling_tester(
        tester_model: Model,
        implementation_name: str,
        implementation: Implementation,
        parent_ports: list[Port],
    ) -> list[Port]:
        """Connect the O_I and S ports of the implementation to the sibling.

        This also nests the sibling in the parent's timeline, next to the
        implementation, by adding a port to parent_ports that connects to the
        sibling's muscle_settings_in.

        Args:
            tester_model: The test model to add conduits and matching timelines to.
            implementation_name: Name of the implementation.
            implementation: The implementation to test.
            parent_ports: The ports of the parent tester, which will be extended.

        Returns:
            The ports of the sibling tester, or an empty list if the implementation
            has no O_I or S ports, in which case no sibling is needed.
        """
        implementation_ports = [
            port
            for port in implementation.ports.values()
            if port.operator in (Operator.O_I, Operator.S)
        ]
        if not implementation_ports:
            return []

        sibling_ports = [
            MuscleTester._connect_mirrored_port(
                tester_model, SIBLING_TESTER_NAME, implementation_name, port
            )
            for port in implementation_ports
        ]

        # Nest the sibling in the parent's timeline, next to the implementation
        tester_model.conduits.append(
            Conduit(
                f"{PARENT_TESTER_NAME}.{SIBLING_SETTINGS_PORT_NAME}",
                f"{SIBLING_TESTER_NAME}.muscle_settings_in",
            )
        )
        parent_ports.append(Port(Identifier(SIBLING_SETTINGS_PORT_NAME), Operator.O_I))

        # And declare the timelines of the mirrored ports to match
        def port_timeline(component: str, timeline: Timeline | None) -> Timeline:
            if timeline is None:
                return Timeline([PARENT_TESTER_NAME, component])
            return Timeline(
                [PARENT_TESTER_NAME, *(f"{component}.{p}" for p in timeline)]
            )

        timelines: list[Timeline | None] = []
        for port in implementation_ports:
            if (port.timeline or None) not in timelines:
                timelines.append(port.timeline or None)

        tester_model.matching_timelines = [
            MatchingTimelines(
                port_timeline(implementation_name, timeline),
                port_timeline(SIBLING_TESTER_NAME, timeline),
            )
            for timeline in timelines
        ]

        return sibling_ports

    @staticmethod
    def _connect_mirrored_port(
        tester_model: Model, tester_name: str, implementation_name: str, port: Port
    ) -> Port:
        """Connect a port of the implementation to a mirrored port of a tester.

        Args:
            tester_model: The test model to add the conduit to.
            tester_name: Name of the tester component.
            implementation_name: Name of the implementation.
            port: The port of the implementation.

        Returns:
            The port of the tester, which sends if the implementation receives and
            vice versa.
        """
        tester_port = f"{tester_name}.{port.name}"
        implementation_port = f"{implementation_name}.{port.name}"
        if port.operator.allows_receiving():
            conduit = Conduit(tester_port, implementation_port)
            tester_operator = Operator.O_I
        else:
            conduit = Conduit(implementation_port, tester_port)
            tester_operator = Operator.S

        tester_model.conduits.append(conduit)
        return Port(port.name, tester_operator, port.timeline)

    def start_implementation(
        self,
        ymmsl_source: str | Path,
        implementation: str,
        *,
        default_timeout: float = 60,
    ) -> ImplementationTester:
        """Start a MUSCLE3 manager and return an ImplementationTester.

        Tester components are added and connected to all ports of the
        implementation defined in the yMMSL source. A subprocess is started in
        which the MUSCLE3 manager runs, and its address is retrieved. A
        monkeypatch overwrites :meth:`ReceiveTimeoutHandler.on_timeout` so that
        a :exc:`RuntimeError` is raised when a receive timeout is reached,
        causing the test simulation to quit. Finally, an
        :class:`ImplementationTester` is created from the manager address and
        the generated test yMMSL configuration.

        Args:
            ymmsl_source: Either a string containing the yMMSL, or a
                :class:`pathlib.Path` pointing to a file containing the yMMSL.
            implementation: Name of the implementation to test.
            default_timeout: Timeout (seconds) for message operations.

        Returns:
            An ImplementationTester connected to the running manager.

        Raises:
            RuntimeError: If the :class:`ImplementationTester` could not be
                initialized, for example because the executable under test does
                not exist and never registered with the manager.
        """
        ymmsl_config = ymmsl.load_as(ymmsl.v0_2.Configuration, ymmsl_source)
        test_ymmsl_config = self._add_tester_component(ymmsl_config, implementation)

        # Save the test configuration to a temporary file
        test_ymmsl_path = self.run_dir / "test_config.ymmsl"
        ymmsl.save(test_ymmsl_config, test_ymmsl_path)

        # The manager needs a flat configuration with resolved timelines
        for model in test_ymmsl_config.models.values():
            resolve_timelines(model)
        test_ymmsl_config = flatten(test_ymmsl_config, Reference(TEST_MODEL_NAME))

        server_ctx = make_server_process(test_ymmsl_config, self.run_dir, True)
        muscle_manager_address = self._exitstack.enter_context(server_ctx)

        # patch ReceiveTimeoutHandler so we can (ab)use it for our timeouts:
        self._exitstack.enter_context(
            patch.object(ReceiveTimeoutHandler, "on_timeout", raise_error)
        )
        self._exitstack.enter_context(
            patch(
                "libmuscle.mcp.tcp_transport_client.RECONNECT_TIMEOUT",
                min(RECONNECT_TIMEOUT, default_timeout),
            )
        )
        self._exitstack.enter_context(
            patch(
                "libmuscle.mmp_client.PEER_TIMEOUT", min(PEER_TIMEOUT, default_timeout)
            )
        )
        # Ensure we won't wait forever on our outboxes
        self._exitstack.enter_context(
            patch("libmuscle.post_office.PostOffice.wait_for_receivers")
        )
        # And we close() our TCP Servers ungracefully
        origclose = TcpTransportServer.close
        self._exitstack.enter_context(
            patch.multiple(
                TcpTransportServer,
                close=lambda self, _=True: origclose(self, False),
            )
        )
        self.implementation_tester = ImplementationTester(
            default_timeout, muscle_manager_address, test_ymmsl_config
        )
        self._exitstack.callback(self.implementation_tester.cleanup)
        return self.implementation_tester

    def cleanup(self) -> None:
        """Stop the manager process and clean up all resources.

        Stops the :class:`ImplementationTester`, restores the monkeypatched
        :meth:`ReceiveTimeoutHandler.on_timeout`, and shuts down the manager
        subprocess.
        """
        self._exitstack.close()
        self.implementation_tester = None


def start_mmp_server(
    control_pipe: tuple[Connection, Connection],
    ymmsl_config: Configuration,
    run_dir: RunDir,
    env: dict[str, str],
    start_instances: bool,
) -> None:
    if start_instances:
        os.environ.clear()
        os.environ.update(env)

    control_pipe[0].close()
    manager = Manager(ymmsl_config, run_dir, "DEBUG")
    control_pipe[1].send(manager.get_server_location())

    if start_instances:
        manager.start_instances()

    control_pipe[1].recv()
    control_pipe[1].close()
    manager.stop()


@contextmanager
def make_server_process(
    ymmsl_config: Configuration, run_dir: Path, start_instances: bool
) -> Generator[str, None, None]:
    run_dir_obj = RunDir(run_dir)
    env = os.environ.copy()
    control_pipe = mp.Pipe()
    process = mp.Process(
        target=start_mmp_server,
        args=(control_pipe, ymmsl_config, run_dir_obj, env, start_instances),
        name="Manager",
    )
    process.start()
    control_pipe[1].close()
    muscle_manager_address = control_pipe[0].recv()
    try:
        yield muscle_manager_address
    finally:
        control_pipe[0].send(True)
        control_pipe[0].close()
        process.join()
