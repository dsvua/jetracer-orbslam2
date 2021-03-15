import { atomFamily, atom } from "recoil";

export const connectionStateIsConnected = atom({
    key: 'connection-state-is-connected-atom',
    default: false,
});

export const connectionStateWs = atom({
    key: 'connection-state-ws-atom',
    default: null,
});

export const connectionStateUrl = atom({
    key: 'connection-state-url-atom',
    default: 'ws://localhost:9002',
});
