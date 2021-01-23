#ifndef JETRACER_THREAD_EVENT_H
#define JETRACER_THREAD_EVENT_H

#include "../Context.h"
#include "BaseEvent.h"

namespace Jetracer
{

    class ThreadEvent : public BaseEvent
    {
    public:
        std::string thread_name;
    };

    typedef std::shared_ptr<ThreadEvent> pThreadEvent; // p - means pointer
} // namespace Jetracer

#endif // JETRACER_THREAD_EVENT_H
