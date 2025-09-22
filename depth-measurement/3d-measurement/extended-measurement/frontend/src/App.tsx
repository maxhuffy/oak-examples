import { css } from "../styled-system/css/css.mjs";
import { Streams, useConnection } from "@luxonis/depthai-viewer-common";
import { ConfidenceSlider } from "./ConfidenceSlider.tsx";
import { ClickCatcher } from "./ClickOverlay.tsx";
import { ClassSelector } from "./ClassSelector.tsx";
import { useRef } from "react";
import { MeasurementMethodSelector } from "./MeasurementMethodSelector";
import { Button } from "@luxonis/common-fe-components";

function App() {
    const connection = useConnection();

    const viewerRef = useRef<HTMLDivElement>(null);

    const selectionService = "Selection Service";

    const clearSelection = () => {
    (connection as any)?.daiConnection?.postToService(selectionService, { clear: true });
    };

    return (
        <main className={css({
            width: 'screen',
            height: 'screen',
            display: 'flex',
            flexDirection: 'row',
            gap: 'md',
            padding: 'md'
        })}>
            {/* Left: Stream Viewer */}
            <div ref={viewerRef} className={css({ flex: 1, position: "relative" })}>
            <Streams
                // pre-open both panels (works after you clear saved selection once)
                defaultTopics={["Video", "Pointclouds"]}
                // optional: limit the picker to just these 2 groups (type wants strings, not booleans)
                topicGroups={{ images: "Images", point_clouds: "Pointclouds" }}
            />
            <ClickCatcher
            containerRef={viewerRef}
            frameWidth={640}
            frameHeight={400}
            debug
            allowedPanelTitle="Video" // or "[aria-label='Images']"
            />
            </div>


            {/* Vertical Divider */}
            <div className={css({
                width: '2px',
                backgroundColor: 'gray.300'
            })} />

            {/* Right: Sidebar (Info and Controls) */}
            <div className={css({
                width: 'md',
                display: 'flex',
                flexDirection: 'column',
                gap: 'md'
            })}>
                <h1 className={css({ fontSize: 'xl', fontWeight: 'bold' })}>
                    Extended 3D Measurement Application
                </h1>
                <p>
                    This example combines a YOLOE segmentation model with DepthAI point clouds to measure real-world objects in 3D.
                    Click any detected object in the Video panel to segment it and get its dimensions and volume.
                </p>

                <ClassSelector />

                {/* Confidence Slider */}
                <ConfidenceSlider initialValue={0.3} />

                <Button
                onClick={clearSelection}         
                alignSelf="start"
                title="Removes the current segmented object. Keeps classes and settings."
                >
                Clear selected object
                </Button>

                <MeasurementMethodSelector />

                {/* Connection Status */}
                <div className={css({
                    display: 'flex',
                    alignItems: 'center',
                    gap: 'xs',
                    marginTop: 'auto',
                    color: connection.connected ? 'green.500' : 'red.500'
                })}>
                    <div className={css({
                        width: '3',
                        height: '3',
                        borderRadius: 'full',
                        backgroundColor: connection.connected ? 'green.500' : 'red.500'
                    })} />
                    <span>{connection.connected ? 'Connected to device' : 'Disconnected'}</span>
                </div>
            </div>
        </main>
    );
}

export default App;
