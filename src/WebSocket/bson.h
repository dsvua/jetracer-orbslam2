#ifndef JETRACER_WEBSOCKETCOM_BSON_H
#define JETRACER_WEBSOCKETCOM_BSON_H

#include <iostream>
#include <string_view>
#include <memory>
#include <vector>

namespace Jetracer
{
    enum class bson_value_type
    {
        bson_double = 0x01,
        bson_int32 = 0x10,
        bson_int64 = 0x11,
        bson_string = 0x02,
        bson_binary = 0x05,
        bson_binary_subtype = 0x80
    };

    typedef struct bson_item
    {
        /* data */
        std::string key; // cstring: str.c_str()
        bson_value_type value_type;
        char *value;
        uint32_t value_size;

        bson_item()
        {
            key = "";
            value_type = bson_value_type::bson_int32;
            value = nullptr;
            value_size = 4;
        };

    } bson_item_t;

    class Bson
    {
    public:
        // Bson();
        ~Bson();

        template <typename T>
        void add(std::string key,
                 bson_value_type value_type,
                 T *value,
                 std::size_t value_bytes = 0)
        {
            bson_item tmp_item;
            // std::cout << "Added item to bson key " << key
            //           << " value_type " << int(value_type)
            //           << " value_bytes " << value_bytes
            //           << std::endl;

            tmp_item.value_type = value_type;
            size_ += 1; // +1 for value type

            // adding key
            size_ += key.size() + 1; // +1 for 0 terminated cstring
            tmp_item.key = key;

            // adding value
            tmp_item.value = reinterpret_cast<char *>(value);
            tmp_item.value_size = value_bytes;
            switch (value_type)
            {
            case bson_value_type::bson_double:
                size_ += sizeof(double);
                tmp_item.value_size = sizeof(double);
                break;
            case bson_value_type::bson_int32:
                size_ += sizeof(int32_t);
                tmp_item.value_size = sizeof(int32_t);
                break;
            case bson_value_type::bson_int64:
                size_ += sizeof(int64_t);
                tmp_item.value_size = sizeof(int64_t);
                break;
            case bson_value_type::bson_string:
                size_ += value_bytes + sizeof(int32_t);
                break;
            case bson_value_type::bson_binary:
                size_ += value_bytes + sizeof(char) + sizeof(int32_t); // +1 for subtype
                break;
            default:
                break;
            }

            items_.push_back(tmp_item);
        };

        // void reset();
        void process();
        uint8_t *ptr();
        // std::shared_ptr<uint8_t[]> get_buffer();
        uint32_t size();

    private:
        // unsigned char *buffer = nullptr;
        // unsigned char *current_pos = nullptr;
        uint32_t size_ = sizeof(uint32_t);
        std::vector<bson_item_t> items_;
        // std::shared_ptr<uint8_t[]> buffer_;
        uint8_t *buffer_ = nullptr;
    };

} // namespace Jetracer

#endif // JETRACER_WEBSOCKETCOM_BSON_H
