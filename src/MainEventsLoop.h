#ifndef JETRACER_MAIN_EVENTS_LOOP_THREAD_H
#define JETRACER_MAIN_EVENTS_LOOP_THREAD_H

#include <iostream>
#include <mutex>
#include <map>

#include "EventsThread.h"
#include "constants.h"
#include "Context.h"
#include "Events/BaseEvent.h"
#include "Events/EventTypes.h"

namespace Jetracer
{

    class MainEventsLoop : public EventsThread
    {
    public:
        MainEventsLoop(const std::string threadName, context_t *ctx);
        // ~MainEventsLoop();

        bool subscribeForEvent(EventType _event_type, std::string _thread_name,
                               std::function<bool(pEvent)> pushEventCallback);
        bool unSubscribeFromEvent(EventType _event_type, std::string _thread_name);

    private:
        void handleEvent(pEvent event);

        context_t *_ctx;
        std::mutex m_mutex_subscribers;
        std::map<EventType, std::map<std::string, std::function<bool(pEvent)>>> _subscribers;
        std::vector<Jetracer::EventsThread *> _started_threads;
    };
} // namespace Jetracer

#endif // JETRACER_MAIN_EVENTS_LOOP_THREAD_H
