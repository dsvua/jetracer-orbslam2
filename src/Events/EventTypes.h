#ifndef JETRACER_THREAD_EVENT_TYPES_H
#define JETRACER_THREAD_EVENT_TYPES_H

#include <iostream>

namespace Jetracer
{

    enum class EventType
    {

        // thread events
        event_start_thread,
        event_stop_thread,

        // keep alive
        event_ping,
        event_pong,
    };

    std::ostream &operator<<(std::ostream &os, EventType &event_type);

} // namespace Jetracer

#endif // JETRACER_THREAD_EVENT_TYPES_H