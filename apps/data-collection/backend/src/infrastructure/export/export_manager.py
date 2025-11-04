from config.system_configuration import SystemConfiguration
from core.model_state import ModelState
from core.snapping.conditions_engine import ConditionsEngine
from config.system_state_exporter import SystemStateExporter
from services.export_service.export_service import ExportService


class ExportManager:
    """
    Facade for the configuration export subsystem.

    Responsibilities:
      • Build SystemStateExporter
      • Create GetConfigService exposing export API
    """

    def __init__(
        self,
        model_state: ModelState,
        condition_engine: ConditionsEngine,
        config: SystemConfiguration,
    ):
        self._model_state = model_state
        self._condition_engine = condition_engine
        self._config = config

        self._exporter: SystemStateExporter = None
        self._service: ExportService = None
        self._build()

    def _build(self):
        self._exporter = SystemStateExporter(self._model_state, self._condition_engine)
        export_service = ExportService(self._exporter)
        self._built = True
        self._register_service(export_service)

    def _register_service(self, export_service: ExportService):
        self._config.visualizer.registerService(
            export_service.get_name(), export_service.handle
        )
