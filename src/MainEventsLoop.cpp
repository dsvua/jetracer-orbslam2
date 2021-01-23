#include "MainEventsLoop.h"
#include "PingPong.h"
#include "RealSense/RealSenseD400.h"
#include <iostream>

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

        std::cout << "Starting PingPong" << std::endl;
        _started_threads.push_back(new Jetracer::PingPong("PingPong", _ctx));
        _started_threads.back()->createThread();

        std::cout << "Starting RealSenseD400" << std::endl;
        _started_threads.push_back(new Jetracer::RealSenseD400("RealSenseD400", _ctx));
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

        switch (event->event_type)
        {
        case EventType::event_stop_thread:
        {
            std::unique_lock<std::mutex> lk(m_mutex);
            std::cout << "MainEventsLoop::handleEvent() EventType::event_stop_thread" << std::endl;
            for (auto started_thread : _started_threads)
            {
                std::cout << "Sending exitThread to " << started_thread->THREAD_NAME << std::endl;
                started_thread->exitThread();
            }
            break;
        }

        default:
        {
            for (auto &subscriber : _subscribers[event->event_type])
            {
                std::function<bool(pEvent)> pushEventToSubscriber = subscriber.second;
                pushEventToSubscriber(event);
            }
            break;
        }
        }
    }

} // namespace Jetracer