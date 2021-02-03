import { useEffect } from 'react';
import { useRecoilState, useSetRecoilState } from 'recoil';
import { connectionStateIsConnected, connectionStateWs } from '../state/network'
import { videoImage, videoImageParams } from '../state/video'
import BSON from 'bson';

export const useWebsocket = () => {

    const setIsConnected = useSetRecoilState(connectionStateIsConnected);
    const setVideoImageParams = useSetRecoilState(videoImageParams);
    const setNewVideoImage = useSetRecoilState(videoImage);
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
                    setNewVideoImage(msg.image.buffer);
                    setVideoImageParams(imageParams);
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