#include "EventTypes.h"

namespace Jetracer
{

    std::ostream &operator<<(std::ostream &os, EventType &event_type)
    {
        switch (event_type)
        {
        case EventType::event_start_thread:
            os << "event_start_thread";
            break;

        case EventType::event_stop_thread:
            os << "event_stop_thread";
            break;

        case EventType::event_ping:
            os << "event_ping";
            break;

        case EventType::event_pong:
            os << "event_pong";
            break;

        default:
            break;
        }

        return os;
    }
} // namespace Jetracer