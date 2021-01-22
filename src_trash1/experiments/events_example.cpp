
class BaseEvent {
public:
    int id;
}

class FrameEvent: public BaseEvent {
public:
    RS::DepthFrame Frame;
}

class NetEvent: public BaseEvent {
public:
    std::string command;
}


class MsgQueue {
private:

    std::queue< std::shared_ptr<BaseEvent> > m_queue;


    std::thread* m_thread;
    std::mutex m_mutex;
    std::condition_variable m_cv;
    const char* THREAD_NAME;
public:
    postEvent(BaseEvent*);
    start(){
        while (true){
            std::shared_ptr<BaseEvent> event = queue.pop();
            switch (event->id){
                case SOME_DEFINED_TYPE_OF_EVENT: {
                    FrameEvent* frame_event = static_cast<FrameEvent*>(event);
                }

            }
        }
    };

};


class fooBar(){
    
    MsgQueue queue = new MsgQueue();

    FrameEvent* event1 = make_shared<FrameEvent>();
    NetEvent* event2 = make_shared<NetEvent>();

    queue.post(event1);


}
