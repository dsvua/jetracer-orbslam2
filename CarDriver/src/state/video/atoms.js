import { atomFamily, atom } from "recoil";

export const videoImage = atom({
    key: 'video-image-atom',
    default: null,
});

export const videoImageParams = atom({
    key: 'video-image-params-atom',
    default: {
        height: 480,
        width: 848,
        channels: 1,
    },
});

export const cameraAngle = atom({
    key: 'camera-angle-atom',
    default: null,
});

