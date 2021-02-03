#include "WebSocketCom.h"
#include "jsoncons/json.hpp"
#include <jsoncons_ext/bson/bson.hpp>

#include <memory>
#include <chrono>
#include <iostream>

#include "../RealSense/RealSenseD400.h"

// using namespace std;

namespace Jetracer
{
    WebSocketCom::WebSocketCom(const std::string threadName, context_t *ctx) : EventsThread(threadName), _ctx(ctx)
    {
        auto pushEventCallback = [this](pEvent event) -> bool {
            this->pushEvent(event);
            return true;
        };

        // _ctx->subscribeForEvent(EventType::event_ping, threadName, pushEventCallback);
        // _ctx->subscribeForEvent(EventType::event_pong, threadName, pushEventCallback);

        CommunicationThread = new std::thread(&WebSocketCom::Communication, this);

        std::cout << "WebSocket is initialized" << std::endl;
    }

    void WebSocketCom::on_message(websocketpp::connection_hdl hdl, server::message_ptr msg)
    {

        auto message_type = msg->get_opcode();
        switch (message_type)
        {
        case websocketpp::frame::opcode::text:
        {
            std::cout << "Text message received:" << std::endl;
            auto j = jsoncons::json::parse(msg->get_payload());
            // std::cout << jsoncons::pretty_print(j) << std::endl;
            break;
        }

        case websocketpp::frame::opcode::binary:
        {
            // std::string message = msg->get_payload();
            // std::vector<uint8_t> buffer = message.data();
            // auto tt = message.data();
            std::cout << "Binary message received:" << std::endl;
            auto j = jsoncons::bson::decode_bson<jsoncons::ojson>(msg->get_payload());
            // std::cout << jsoncons::pretty_print(j) << std::endl;
            break;
        }

        default:
            break;
        }
    }

    void WebSocketCom::on_open(connection_hdl hdl)
    {
        m_connections.insert(hdl);
    }

    void WebSocketCom::on_close(connection_hdl hdl)
    {
        m_connections.erase(hdl);
    }

    void WebSocketCom::send_message()
    {
        // Broadcast message to all connections
        // if (m_endpoint.is_listening())
        // {
        //     con_list::iterator it;
        //     for (it = m_connections.begin(); it != m_connections.end(); ++it)
        //     {
        //         m_endpoint.send(*it, "some very important message", websocketpp::frame::opcode::text);
        //     }
        // }
    }

    void WebSocketCom::Communication()
    {
        auto pushEventCallback = [this](pEvent event) -> bool {
            this->pushEvent(event);
            return true;
        };

        _ctx->subscribeForEvent(EventType::event_realsense_D400_rgbd, THREAD_NAME, pushEventCallback);

        using websocketpp::lib::bind;
        using websocketpp::lib::placeholders::_1;
        using websocketpp::lib::placeholders::_2;
        m_endpoint.clear_access_channels(websocketpp::log::alevel::all);
        m_endpoint.set_message_handler(bind(&WebSocketCom::on_message, this, _1, _2));
        m_endpoint.set_open_handler(bind(&WebSocketCom::on_open, this, _1));
        m_endpoint.set_close_handler(bind(&WebSocketCom::on_close, this, _1));

        m_endpoint.init_asio();
        m_endpoint.listen(_ctx->websocket_port);
        m_endpoint.start_accept();
        // m_endpoint.set_timer();
        std::cout << "Server Started." << std::endl;

        // Start the ASIO io_service run loop
        try
        {
            m_endpoint.run();
        }
        catch (websocketpp::exception const &e)
        {
            std::cout << e.what() << std::endl;
        }

        // to send message
        // server::connection_ptr con = m_endpoint.get_con_from_hdl(hdl);
        // std::string resp("BAD");
        // con->send(resp, websocketpp::frame::opcode::text);

        std::cout << "Exiting WebSocket::Communication" << std::endl;
    }

    void WebSocketCom::handleEvent(pEvent event)
    {

        switch (event->event_type)
        {

        case EventType::event_stop_thread:
        {
            std::cout << "Stopping CommunicationThread" << std::endl;
            m_endpoint.stop_listening();
            m_endpoint.stop();
            CommunicationThread->join();
            std::cout << "Stopped CommunicationThread" << std::endl;
            break;
        }

        case EventType::event_realsense_D400_rgbd:
        {
            std::shared_ptr<rgbd_frame_t> rgbd_frame = std::static_pointer_cast<rgbd_frame_t>(event->message);
            std::size_t image_size = _ctx->cam_w * _ctx->cam_w;
            const char *char_data = static_cast<const char *>(rgbd_frame->lefr_ir);
            std::basic_string_view view_data(char_data, image_size);

            // std::cout << "Creating bson message" << std::endl;
            std::vector<uint8_t> buffer;
            jsoncons::bson::bson_bytes_encoder encoder(buffer);
            encoder.begin_object();

            encoder.key("timestamp");
            encoder.double_value(rgbd_frame->timestamp);
            encoder.key("width");
            encoder.uint64_value(_ctx->cam_w);
            encoder.key("height");
            encoder.uint64_value(_ctx->cam_h);
            encoder.key("channels");
            encoder.uint64_value(1);
            encoder.key("image");
            encoder.byte_string_value(view_data);

            encoder.end_object();
            encoder.flush();

            con_list::iterator it;
            for (it = m_connections.begin(); it != m_connections.end(); ++it)
            {
                // std::cout << "Sending bson message" << std::endl;
                auto con = m_endpoint.get_con_from_hdl(*it);
                if (con->get_buffered_amount() < _ctx->cam_w * _ctx->cam_h * _ctx->WebSocketCom_max_queue_legth)
                {
                    try
                    {
                        m_endpoint.send(*it, buffer.data(), buffer.size(), websocketpp::frame::opcode::binary);
                        // std::cout << "Bson message is sent" << std::endl;
                    }
                    catch (websocketpp::exception const &e)
                    {
                        std::cout << e.what() << std::endl;
                    }
                }
                else
                {
                    std::cout << "Cannot send, buffer is " << con->get_buffered_amount() << std::endl;
                }
            }
            break;
        }

        default:
        {
            // std::cout << "Got unknown message of type " << event->event_type << std::endl;
            break;
        }
        }
    }

} // namespace Jetracer