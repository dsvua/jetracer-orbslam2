import { selectorFamily, selector } from "recoil";
import { connectionState } from "./atoms";


export const connectionConnectSelector = selector({
    key: 'connection-connect',
    set: ({set, get}, url) => {
        
    }
})