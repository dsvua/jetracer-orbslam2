
#include "EventsThread.h"
#include <iostream>

namespace Jetracer
{
    EventsThread::EventsThread(const std::string threadName) : THREAD_NAME(threadName), m_thread(0)
    {
    }

    EventsThread::~EventsThread()
    {
        exitThread();
    }

    bool EventsThread::createThread()
    {
        if (!m_thread)
            m_thread = new std::thread(&EventsThread::process, this);
        return true;
    }

    std::thread::id EventsThread::getThreadId()
    {
        ASSERT_TRUE(m_thread != 0);
        return m_thread->get_id();
    }

    std::thread::id EventsThread::getCurrentThreadId()
    {
        return std::this_thread::get_id();
    }

    void EventsThread::exitThread()
    {
        if (!m_thread)
            return;

        quit_thread = true;

        pEvent event = std::make_shared<BaseEvent>();
        event->event_type = EventType::event_stop_thread;

        std::cout << "Sent event_stop_thread event, prepare to join" << std::endl;
        // Put exit thread message into the queue
        std::lock_guard<std::mutex> lock(m_mutex);
        m_queue.push(event);
        m_mutex.unlock();
        m_cv.notify_one();

        std::cout << "Joining threads in " << THREAD_NAME << std::endl;
        m_thread->join();
        delete m_thread;
        m_thread = 0;
    }

    bool EventsThread::pushEvent(pEvent event)
    {
        // std::cout << "Got message, adding to queue" << std::endl;
        // Add event to queue and notify process method
        std::unique_lock<std::mutex> lk(m_mutex);
        // prevent queue explosion when events processed slower then events coming in
        if (event->event_type == EventType::event_stop_thread || m_queue.size() < max_queue_length)
        {
            m_queue.push(event);
            m_cv.notify_one();
        }
        else
        {
            // std::cout << "Skipping message in " << THREAD_NAME << " thread, total messages: " << m_queue.size() << std::endl;
        }

        return true;
    }

    void EventsThread::setMaxQueueLength(int new_max_queue_length)
    {
        max_queue_length = new_max_queue_length;
    }

    void EventsThread::process()
    {
        while (!quit_thread)
        {
            // Wait for a message to be added to the queue
            std::unique_lock<std::mutex> lk(m_mutex);
            while (m_queue.empty())
                m_cv.wait(lk);

            if (m_queue.empty())
                continue;

            pEvent event = m_queue.front();
            m_queue.pop();

            handleEvent(event);

            switch (event->event_type)
            {
            case EventType::event_stop_thread:
            {
                quit_thread = true;
                while (!m_queue.empty())
                {
                    event = m_queue.front();
                    m_queue.pop();
                }
                std::cout << "Exited process for thread " << THREAD_NAME << std::endl;
                return;
            }

            default:
                break;
            }
        }
    }
} // namespace Jetracer