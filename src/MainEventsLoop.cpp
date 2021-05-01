#include "MainEventsLoop.h"
#include "PingPong/PingPong.h"
#include "RealSense/RealSenseD400.h"
#include "RealSense/SaveRawData.h"
#include "WebSocket/WebSocketCom.h"
#include "SlamGpuPipeline/SlamGpuPipeline.h"

#include <iostream>
#include <unistd.h> // for sleep function

// #include <memory>

namespace Jetracer
{

    MainEventsLoop::MainEventsLoop(const std::string threadName, context_t *ctx) : EventsThread(threadName), _ctx(ctx)
    {
        // pushing callbacks
        _ctx->sendEvent = [this](pEvent event) -> bool {
            this->pushEvent(event);
            return true;
        };

        _ctx->subscribeForEvent = [this](EventType _event_type,
                                         std::string _thread_name,
                                         std::function<bool(pEvent)> _pushEventCallback) -> bool {
            this->subscribeForEvent(_event_type, _thread_name, _pushEventCallback);
            return true;
        };

        _ctx->unSubscribeFromEvent = [this](EventType _event_type,
                                            std::string _thread_name) -> bool {
            this->unSubscribeFromEvent(_event_type, _thread_name);
            return true;
        };

        // std::cout << "Starting PingPong" << std::endl;
        // _started_threads.push_back(new Jetracer::PingPong("PingPong", _ctx));
        // _started_threads.back()->setMaxQueueLength(_ctx->PingPong_max_queue_legth);
        // _started_threads.back()->createThread();

        std::cout << "Starting RealSenseD400" << std::endl;
        _started_threads.push_back(new Jetracer::RealSenseD400("RealSenseD400", _ctx));
        _started_threads.back()->setMaxQueueLength(_ctx->RealSenseD400_max_queue_legth);
        _started_threads.back()->createThread();

        // std::cout << "Starting SaveRawData" << std::endl;
        // _started_threads.push_back(new Jetracer::SaveRawData("SaveRawData", _ctx));
        // _started_threads.back()->setMaxQueueLength(_ctx->SaveRawData_max_queue_legth);
        // _started_threads.back()->createThread();

        // std::cout << "Starting WebSocket" << std::endl;
        _started_threads.push_back(new Jetracer::WebSocketCom("WebSocketCom", _ctx));
        _started_threads.back()->setMaxQueueLength(_ctx->WebSocketCom_max_queue_legth);
        _started_threads.back()->createThread();

        // std::cout << "Starting SlamGpuPipeline" << std::endl;
        _started_threads.push_back(new Jetracer::SlamGpuPipeline("SlamGpuPipeline", _ctx));
        _started_threads.back()->setMaxQueueLength(_ctx->SlamGpuPipeline_max_queue_length);
        _started_threads.back()->createThread();
    }

    // MainEventsLoop::~MainEventsLoop()
    // {
    // }

    bool MainEventsLoop::subscribeForEvent(EventType _event_type,
                                           std::string _thread_name,
                                           std::function<bool(pEvent)> pushEventCallback)
    {
        std::unique_lock<std::mutex> lk(m_mutex_subscribers);
        _subscribers[_event_type][_thread_name] = pushEventCallback;
        return true;
    }

    bool MainEventsLoop::unSubscribeFromEvent(EventType _event_type,
                                              std::string _thread_name)
    {
        std::unique_lock<std::mutex> lk(m_mutex_subscribers);
        _subscribers[_event_type].erase(_thread_name);
        return true;
    }

    void MainEventsLoop::handleEvent(pEvent event)
    {

        // std::cout << "MainEventsLoop distribute event " << event->event_type << std::endl;

        for (auto &subscriber : _subscribers[event->event_type])
        {
            // std::function<bool(pEvent)> pushEventToSubscriber = subscriber.second;
            // pushEventToSubscriber(event);
            subscriber.second(event);
        }

        // std::cout << "MainEventsLoop handle switch event " << event->event_type << std::endl;

        switch (event->event_type)
        {
        case EventType::event_stop_thread:
        {
            std::cout << "MainEventsLoop::handleEvent() EventType::event_stop_thread" << std::endl;
            for (auto running_thread : _started_threads)
            {
                std::cout << "Sending exitThread to " << running_thread->THREAD_NAME << std::endl;
                running_thread->exitThread();
            }
            std::cout << "All threads exited " << std::endl;
            break;
        }

        default:
        {
            // std::cout << "Got event in MainLoop " << event->event_type << std::endl;
            break;
        }
        }
    }

} // namespace Jetracer