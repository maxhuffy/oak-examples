import depthai as dai

from config.system_configuration import SystemConfiguration
from core.model_state import ModelState
from infrastructure.neural_network.neural_network_manager import NeuralNetworkManager
from infrastructure.snaps.snaps_manager import SnapsManager
from infrastructure.system_context import SystemContext
from infrastructure.video_source_manager import VideoSourceManager
from infrastructure.export.export_manager import ExportManager


def main():
    system_context = SystemContext()
    config = SystemConfiguration(system_context.platform)

    with dai.Pipeline(system_context.device) as pipeline:
        print("Creating pipeline...")

        video_source = VideoSourceManager(pipeline, config.get_video_config())

        system_context.add_visualizer_topic(video_source.get_video_topic(), "Video")

        model_state = ModelState()

        nn_manager = NeuralNetworkManager(
            pipeline, video_source, config.get_neural_network_config(), model_state
        )

        system_context.register_services(nn_manager.get_services())

        system_context.add_visualizer_topic(nn_manager.get_annotations(), "Annotations")

        snaps_manager = SnapsManager(
            pipeline, video_source, nn_manager, config.get_snaps_config()
        )

        system_context.register_service(snaps_manager.get_service())

        export_manager = ExportManager(model_state, snaps_manager.get_engine())

        system_context.register_service(export_manager.get_service())

        print("Pipeline created.")
        pipeline.start()
        system_context.register_pipeline(pipeline)

        while pipeline.isRunning():
            pipeline.processTasks()


if __name__ == "__main__":
    main()
