import React, { useEffect, useRef } from 'react';
import { useRecoilValue } from 'recoil';
import { showVideoState } from '../state/network'
import {VideoFeed} from './videofeed'
import {CameraPos} from './camerapos'

export const Feed = (props) => {

    const showVideo = useRecoilValue(showVideoState);

    return (
        <div>
            {showVideo ? <VideoFeed /> : <CameraPos />}
        </div>
    )
}