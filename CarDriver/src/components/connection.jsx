import React, { useState, useEffect } from 'react';
import { useRecoilState, useRecoilValue } from 'recoil';
import { connectionStateUrl, connectionStateIsConnected, connectionStateWs } from '../state/network'

export const WsConnection = () => {

    const [connUrl, setConnUrl] = useRecoilState(connectionStateUrl);
    const isConnected = useRecoilValue(connectionStateIsConnected);
    const [ws, setWs] = useRecoilState(connectionStateWs);

    const handleUrlChange = (e) => {
        setConnUrl(e.target.value);
    };

    const handleConnect = (e) => {

        if(!isConnected) {
            setWs(new WebSocket(connUrl));
        } else {
            ws.close();
        };
    };

    const handleSendTextMessage = (e) => {
        ws.send('{"message": "test message"}');
    };

    return (
        // <div>
        //     <div className="container w-100">
                <div className="row justify-content-center align-items-end">
                    <div className="col justify-content-center">
                        <input type="text" value={connUrl} onChange={handleUrlChange}>
                        </input>
                    </div>
                    <div className="col justify-content-center">
                        <button onClick={handleConnect}
                        type="button"
                        className={isConnected ? "btn btn-warning" : "btn btn-success"}> 
                            {isConnected ? "Disconnect" : "Connect"}
                        </button>
                    </div>
                    <div className="col justify-content-center">
                        <button disabled={!isConnected} onClick={handleSendTextMessage}
                        type="button"
                        className="btn btn-secondary"> 
                            Send Text message
                        </button>
                    </div>
                </div>
        //     </div>
        // </div>
    )
}