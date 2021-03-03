#include "PingPong.h"

#include <memory>
#include <chrono>
#include <iostream>

// using namespace std;

namespace Jetracer
{
    PingPong::PingPong(const std::string threadName, context_t *ctx) : EventsThread(threadName), _ctx(ctx)
    {
        auto pushEventCallback = [this](pEvent event) -> bool {
            this->pushEvent(event);
            return true;
        };

        _ctx->subscribeForEvent(EventType::event_ping, threadName, pushEventCallback);
        _ctx->subscribeForEvent(EventType::event_pong, threadName, pushEventCallback);

        m_timerExit = false;
        timerThread = new std::thread(&PingPong::TimerThread, this);

        std::cout << "PingPong is initialized" << std::endl;
    }

    void PingPong::TimerThread()
    {
        while (!m_timerExit)
        {
            // Sleep for 250ms then put a MSG_TIMER message into queue
            std::this_thread::sleep_for(std::chrono::milliseconds(250));

            // Add timer msg to queue and notify worker thread
            pEvent event = std::make_shared<BaseEvent>();
            event->event_type = EventType::event_ping;
            std::cout << "Sending ping message from timer when m_timerExit=" << m_timerExit << std::endl;
            if (!m_timerExit)
                _ctx->sendEvent(event);
        }
        std::cout << "Exiting timer" << std::endl;
    }

    void PingPong::handleEvent(pEvent event)
    {

        switch (event->event_type)
        {
        case EventType::event_ping:
        {
            std::cout << "Got ping message" << std::endl;

            // Sending pong message
            pEvent ping_event = std::make_shared<BaseEvent>();
            ping_event->event_type = EventType::event_pong;
            _ctx->sendEvent(ping_event);
            break;
        }

        case EventType::event_pong:
        {
            std::cout << "Got pong message" << std::endl;
            break;
        }

        case EventType::event_stop_thread:
        {
            std::cout << "joining timerThread" << std::endl;
            m_timerExit = true;
            timerThread->join();
            std::cout << "timerThread joined" << std::endl;
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