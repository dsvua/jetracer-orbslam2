#ifndef JETRACER_WEBSOCKETCOM_THREAD_H
#define JETRACER_WEBSOCKETCOM_THREAD_H

#include <iostream>

#include "../EventsThread.h"
#include "../Context.h"
#include "../Events/BaseEvent.h"
#include "../Events/EventTypes.h"
#include <mutex>
#include <atomic>
#include <thread>
#include <set>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudafeatures2d.hpp>

#define ASIO_STANDALONE
#include <websocketpp/config/asio_no_tls.hpp>
#include <websocketpp/server.hpp>

typedef websocketpp::server<websocketpp::config::asio> server;
typedef server::message_ptr message_ptr;
typedef websocketpp::connection_hdl connection_hdl;
typedef std::set<connection_hdl, std::owner_less<connection_hdl>> con_list;

namespace Jetracer
{

    class WebSocketCom : public EventsThread
    {
    public:
        WebSocketCom(const std::string threadName, context_t *ctx);
        // ~WebSocketCom();

    private:
        void handleEvent(pEvent event);
        void Communication();
        void on_message(websocketpp::connection_hdl hdl, server::message_ptr msg);
        void on_open(connection_hdl hdl);
        void on_close(connection_hdl hdl);
        void send_message();

        context_t *_ctx;
        std::mutex m_mutex_subscribers;
        server m_endpoint;
        con_list m_connections;
        // cv::Ptr<cv::cuda::ORB> detector;

        std::thread *CommunicationThread;
    };

} // namespace Jetracer

#endif // JETRACER_WEBSOCKETCOM_THREAD_H
