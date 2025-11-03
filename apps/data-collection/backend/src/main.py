import depthai as dai

from config.system_configuration import SystemConfiguration
from core.model_state import ModelState
from infrastructure.neural_network.neural_network_manager import NeuralNetworkManager
from infrastructure.snaps.snaps_manager import SnapsManager
from infrastructure.video_source_manager import VideoSourceManager
from infrastructure.export.export_manager import ExportManager


def main():
    config = SystemConfiguration()
    runtime = config.build_runtime_config()

    with dai.Pipeline(config.device) as pipeline:
        print("Creating pipeline...")

        video_source = VideoSourceManager(pipeline, config, runtime)

        model_state = ModelState()

        nn_manager = NeuralNetworkManager(
            pipeline, video_source, runtime, config, model_state
        )

        snaps_manager = SnapsManager(pipeline, video_source, nn_manager, config)

        ExportManager(model_state, snaps_manager.get_engine(), config)

        print("Pipeline created.")
        pipeline.start()
        config.visualizer.registerPipeline(pipeline)

        print("Press 'q' to stop")

        while pipeline.isRunning():
            pipeline.processTasks()
            key = config.visualizer.waitKey(1)
            if key == ord("q"):
                print("Got q key. Exiting...")
                break


if __name__ == "__main__":
    main()
