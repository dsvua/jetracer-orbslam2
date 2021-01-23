#ifndef JETRACER_BASE_EVENT_H
#define JETRACER_BASE_EVENT_H

#include "EventTypes.h"
#include <memory>

namespace Jetracer
{
    typedef std::shared_ptr<void> pMessage;

    class BaseEvent
    {
    public:
        EventType event_type;
        pMessage message;
    };

    typedef std::shared_ptr<BaseEvent> pEvent;

} // namespace Jetracer

#endif // JETRACER_BASE_EVENT_H