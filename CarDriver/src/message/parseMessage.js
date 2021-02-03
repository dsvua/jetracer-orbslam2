import BSON from 'bson';

export const parseMessage = (message) => {
    
    try {
        const reader = new FileReader()
        reader.onload = function () {
          const msgarray = new Uint8Array(this.result)
          const msg = BSON.deserialize(msgarray)
          console.log(`Received message`);
          console.log(msg.timestamp);
          };

        reader.readAsArrayBuffer(message.data);

      }
      catch (e) {
        console.log(`Failed to deserialise websocket message`)
      }
}