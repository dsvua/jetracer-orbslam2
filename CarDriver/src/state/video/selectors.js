import { selector } from "recoil";
import { videoImage } from "./atoms";


export const newVideoImage = selector({
    key: 'set-new-video-image',
    set: ({set}, newImage) => {
        set(videoImage, newImage);
    }
})