#include "RealSenseD400.h"

#include <memory>
#include <chrono>
#include <iostream>

// using namespace std;

namespace Jetracer
{
    RealSenseD400::RealSenseD400(const std::string threadName, context_t *ctx) : EventsThread(threadName), _ctx(ctx)
    {
        auto pushEventCallback = [this](pEvent event) -> bool {
            this->pushEvent(event);
            return true;
        };

        _ctx->subscribeForEvent(EventType::event_ping, threadName, pushEventCallback);
        _ctx->subscribeForEvent(EventType::event_pong, threadName, pushEventCallback);

        std::cout << "RealSenseD400 is initialized" << std::endl;
    }

    void RealSenseD400::handleEvent(pEvent event)
    {

        switch (event->event_type)
        {

        case EventType::event_stop_thread:
        {
            break;
        }

        default:
        {
            std::cout << "Got unknown message of type " << event->event_type << std::endl;
            break;
        }
        }
    }

} // namespace Jetracer