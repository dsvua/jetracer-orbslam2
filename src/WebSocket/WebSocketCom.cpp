#include "WebSocketCom.h"
#include "jsoncons/json.hpp"
#include <jsoncons_ext/bson/bson.hpp>

#include <memory>
#include <chrono>
#include <iostream>

#include <chrono>
using namespace std::chrono;

#include "../RealSense/RealSenseD400.h"
#include "../SlamGpuPipeline/SlamGpuPipeline.h"
#include "bson.h"
#include "../cuda_common.h"

// using namespace std;

namespace Jetracer
{
    WebSocketCom::WebSocketCom(const std::string threadName, context_t *ctx) : EventsThread(threadName), _ctx(ctx)
    {
        auto pushEventCallback = [this](pEvent event) -> bool {
            this->pushEvent(event);
            return true;
        };

        _ctx->subscribeForEvent(EventType::event_stop_thread, threadName, pushEventCallback);
        _ctx->subscribeForEvent(EventType::event_gpu_slam_frame, threadName, pushEventCallback);

        CommunicationThread = new std::thread(&WebSocketCom::Communication, this);
        current_send_quota = _ctx->WebSocketCom_max_send_rate;
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
            std::cout << "Binary message received:" << std::endl;
            auto j = jsoncons::bson::decode_bson<jsoncons::ojson>(msg->get_payload());
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

        using websocketpp::lib::bind;
        using websocketpp::lib::placeholders::_1;
        using websocketpp::lib::placeholders::_2;
        m_endpoint.set_reuse_addr(true);
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
        // std::cout << "WebSocketCom::handleEvent Got event of type " << event->event_type << std::endl;

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

        case EventType::event_gpu_slam_frame:
        {
            auto entrance = high_resolution_clock::now();

            // std::cout << "--------> WebSocketCom -----" << std::endl;
            std::shared_ptr<slam_frame_t> slam_frame = std::static_pointer_cast<slam_frame_t>(event->message);
            std::shared_ptr<rgbd_frame_t> rgbd_frame = slam_frame->rgbd_frame;
            std::size_t image_size = _ctx->cam_w * _ctx->cam_h * sizeof(char);

            auto start = high_resolution_clock::now();

            auto time_delta = duration_cast<milliseconds>(start - prev_sent_time);
            prev_sent_time = start;

            current_send_quota += _ctx->WebSocketCom_max_send_rate / 1000.0f * time_delta.count();
            if (current_send_quota > _ctx->WebSocketCom_max_send_rate)
                current_send_quota = _ctx->WebSocketCom_max_send_rate;
            // std::cout << "time_delta: " << time_delta.count() << std::endl;

            Bson bson_message;
            // std::cout << "--------> bson_message.add -----" << std::endl;
            int channels = 1;
            int32_t ax = floor(slam_frame->theta.x * 180 / CUDART_PI_D);
            int32_t ay = floor(slam_frame->theta.y * 180 / CUDART_PI_D);
            int32_t az = floor((slam_frame->theta.z - CUDART_PI_D / 2) * 180 / CUDART_PI_D);
            bson_message.add("ax", bson_value_type::bson_int32, &ax);
            bson_message.add("ay", bson_value_type::bson_int32, &ay);
            bson_message.add("az", bson_value_type::bson_int32, &az);
            bson_message.add("width", bson_value_type::bson_int32, &_ctx->cam_w);
            bson_message.add("height", bson_value_type::bson_int32, &_ctx->cam_h);
            bson_message.add("channels", bson_value_type::bson_int32, &channels);
            bson_message.add("keypoints_x",
                             bson_value_type::bson_binary,
                             slam_frame->keypoints_x,
                             slam_frame->h_matched_keypoints_num * sizeof(uint16_t));
            bson_message.add("keypoints_y",
                             bson_value_type::bson_binary,
                             slam_frame->keypoints_y,
                             slam_frame->h_matched_keypoints_num * sizeof(uint16_t));
            bson_message.add("image",
                             bson_value_type::bson_binary,
                             slam_frame->image,
                             slam_frame->image_length * sizeof(char));

            // std::cout << "--------> bson_message.process -----" << std::endl;
            bson_message.process();
            // std::cout << "--------> bson_message.process - done -----" << std::endl;

            auto stop = high_resolution_clock::now();
            auto duration = duration_cast<milliseconds>(stop - start);

            // std::cout << "total number of keypoints: " << slam_frame->keypoints.size()
            //           << " frame_id " << slam_frame->rgbd_frame->original_frame.get_frame_number()
            //           << " BSON creation time " << duration.count() << "ms"
            //           << " image_size " << image_size
            //   << " view_data_x.length() " << view_data_x.length()
            //   << " view_keypoints_x.size() " << view_keypoints_x.size()
            //   << " sizeof(float) " << sizeof(float)
            //   << " Camera->GPU " << duration_cast<milliseconds>(rgbd_frame->GPU_scheduled - rgbd_frame->RS400_callback).count()
            //   << " GPU -> callback " << duration_cast<milliseconds>(rgbd_frame->GPU_callback - rgbd_frame->GPU_scheduled).count()
            //   << " callback -> Websoc Ent " << duration_cast<milliseconds>(entrance - rgbd_frame->GPU_callback).count()
            //   << " callback -> callback Event " << duration_cast<milliseconds>(rgbd_frame->GPU_EventSent - rgbd_frame->GPU_callback).count()
            //   << " callback Event -> Websoc Ent " << duration_cast<milliseconds>(entrance - rgbd_frame->GPU_EventSent).count()
            //   << " Camera -> Websoc Ent " << duration_cast<milliseconds>(entrance - rgbd_frame->GPU_scheduled).count()
            //   << std::endl;

            // std::cout << duration.count() << std::endl;

            // con_list::iterator it;
            // std::cout << "bson_message.size() " << bson_message.size()
            //           << " current_send_quota " << current_send_quota
            //           << std::endl;
            if (bson_message.size() < current_send_quota)
            {
                current_send_quota -= bson_message.size();
                // if (current_send_quota < 0)
                //     current_send_quota = 0;

                for (con_list::iterator it = m_connections.begin(); it != m_connections.end(); ++it)
                {
                    // std::cout << "Sending bson message" << std::endl;
                    auto con = m_endpoint.get_con_from_hdl(*it);
                    if (con->get_buffered_amount() < _ctx->cam_w * _ctx->cam_h * _ctx->WebSocketCom_max_queue_legth)
                    {
                        try
                        {
                            m_endpoint.send(*it, bson_message.ptr(), bson_message.size(), websocketpp::frame::opcode::binary);
                            // m_endpoint.send(*it, buffer.data(), buffer.size(), websocketpp::frame::opcode::binary);
                            std::cout << "Bson message is sent" << std::endl;
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

    Bson::~Bson()
    {
        if (buffer_)
            delete buffer_;
    }

} // namespace Jetracer