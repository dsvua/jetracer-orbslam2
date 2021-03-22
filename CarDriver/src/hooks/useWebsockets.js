import { useEffect } from 'react';
import { useRecoilState, useSetRecoilState } from 'recoil';
import { connectionStateIsConnected, connectionStateWs } from '../state/network'
import { videoImage, videoImageParams, cameraAngle } from '../state/video'
import BSON from 'bson';

export const useWebsocket = () => {

    const setIsConnected = useSetRecoilState(connectionStateIsConnected);
    const setVideoImageParams = useSetRecoilState(videoImageParams);
    const setNewVideoImage = useSetRecoilState(videoImage);
    const setNewAngle = useSetRecoilState(cameraAngle);
    const [ws, setWs] = useRecoilState(connectionStateWs);

    useEffect( () => {
        if(ws) {
            ws.binaryType = "blob";

            ws.onopen = (e) => {
                setIsConnected(true);
                console.log("WebSocket is connected.");
            };

            ws.onclose = (e) => {
                setIsConnected(false);
                console.log("WebSocket is disconnected.");
                setWs(null);
            };

            ws.onmessage = (message) => {
                const reader = new FileReader()
                reader.onload = function () {
                    const msgarray = new Uint8Array(this.result)
                    const msg = BSON.deserialize(msgarray)
                    // console.log(`Received message`);
                    // console.log(msg.timestamp);

                    let imageParams = {};
                    imageParams.height = msg.height;
                    imageParams.width = msg.width;
                    imageParams.channels = msg.channels;

                    // console.log("image pixel:", msg);
                    if (msg.image && msg.keypoints_x && msg.keypoints_y)
                    {
                        setNewVideoImage({
                            image: new Uint8Array(msg.image.buffer),
                            x: Uint8Array.from(msg.keypoints_x.buffer),
                            y: Uint8Array.from(msg.keypoints_y.buffer)
                        });
                        setVideoImageParams(imageParams);
                        // setNewSlamKeyPoints({x: msg.keypoints_x.buffer, y: msg.keypoints_y.buffer})
                    };
                    // console.log("msg.ax && msg.ay && msg.az", msg.ax, msg.ay, msg.az);
                    if (msg.ax && msg.ay && msg.az)
                    {
                        setNewAngle({
                            ax: msg.ax,
                            ay: msg.ay,
                            az: msg.az,
                        });
                    };
                };
                try {
                    reader.readAsArrayBuffer(message.data);
                }
                catch (e) {
                    // console.log(`Failed to deserialise websocket message`);
                    console.log(e);
                }
            };

            ws.onerror = (err) => {
                console.error("WebSocket error observed:", err);
            };
        };

    },[ws]);

    return null
}

export default useWebsocket;