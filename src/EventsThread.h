#include "constants.h"
#include "Context.h"
#include "Events/BaseEvent.h"
#include "Fault.h"
#include <condition_variable>
#include <queue>
#include <thread>

#ifndef JETRACER_EVENTS_THREAD_H
#define JETRACER_EVENTS_THREAD_H

namespace Jetracer
{

    class EventsThread
    {
    public:
        /// Constructor
        EventsThread(const std::string threadName);

        /// Destructor
        ~EventsThread();

        /// Called once to create the worker thread
        /// @return TRUE if thread is created. FALSE otherwise.
        bool createThread();

        /// Called once a program exit to exit the worker thread
        void exitThread();

        /// Get the ID of this thread instance
        /// @return The worker thread ID
        std::thread::id getThreadId();

        /// Get the ID of the currently executing thread
        /// @return The current thread ID
        static std::thread::id getCurrentThreadId();

        /// Add a event to thread queue.
        /// @param[in] event - thread specific information created on the heap using operator new.
        bool pushEvent(pEvent event);

        /// set max_queue_length.
        /// @param[in] max_queue_length - thread specific information created on the heap using operator new.
        void setMaxQueueLength(int new_max_queue_length);

        std::string THREAD_NAME;
        std::thread *m_thread;
        std::queue<pEvent> m_queue;
        std::mutex m_mutex;
        std::condition_variable m_cv;

    protected:
        virtual void handleEvent(pEvent event){};

    private:
        EventsThread(const EventsThread &);
        EventsThread &operator=(const EventsThread &);
        int max_queue_length = 10;
        bool quit_thread = false;

        /// Entry point for the worker thread
        void process();
    };

} // namespace Jetracer

#endif // JETRACER_EVENTS_THREAD_H
