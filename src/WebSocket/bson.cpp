#include "bson.h"
#include <type_traits>
#include <cstring>
#include <sstream>
#include <stdio.h> //for printf

namespace Jetracer
{
    // Bson::Bson()
    // {
    //     // buffer = (unsigned char *)malloc(5000000); // ~5Mb
    // }

    // template <typename T>
    // void Bson::add(std::string key,
    //                bson_value_type value_type,
    //                T value,
    //                std::size_t value_bytes)
    // {
    //     bson_item_t tmp_item;

    //     // adding key
    //     size_ += key.size() + 1;
    //     tmp_item.key = key.c_str();
    //     tmp_item.key_size = key.size() + 1;

    //     // adding value
    //     tmp_item.value_type = value_type;

    //     if (std::is_pointer<T>::value)
    //     {
    //         tmp_item.value = value;
    //         tmp_item.value_size = value_bytes;
    //         size_ += value_bytes;
    //     }
    //     else
    //     {
    //         tmp_item.value = &value;
    //         tmp_item.value_size = sizeof(T);
    //         size_ += sizeof(T);
    //     }

    //     items_.push_back(tmp_item);
    // }

    void Bson::process()
    {
        size_ += 1; // +1 for trailing 0x00
        // buffer_ = std::make_shared<unsigned char[]>(size_);
        if (!buffer_)
        {
            buffer_ = (uint8_t *)malloc(size_ * sizeof(uint8_t));
        }
        // buffer_[408750] = 0;

        uint32_t *tmp_ptr = reinterpret_cast<uint32_t *>(buffer_);
        tmp_ptr[0] = size_;
        int buffer_idx = sizeof(uint32_t);

        for (auto &item : items_)
        {
            // std::cout << "bson processing " << item.key
            //           << " " << item.value_size
            //           << " size_ " << size_
            //           << " buffer_idx " << buffer_idx
            //           << std::endl;

            buffer_[buffer_idx] = (unsigned char)item.value_type;
            buffer_idx += sizeof(char);
            std::memcpy(buffer_ + buffer_idx, item.key.c_str(), item.key.size() + 1);
            buffer_idx += item.key.size() + 1;
            // tmp_ptr = reinterpret_cast<uint32_t *>(buffer_.get() + buffer_idx);
            // tmp_ptr[0] = item.value_size;
            // buffer_idx += sizeof(uint32_t);
            switch (item.value_type)
            {
            case bson_value_type::bson_double:
            case bson_value_type::bson_int32:
            case bson_value_type::bson_int64:
                // write number
                std::memcpy(buffer_ + buffer_idx, item.value, item.value_size);
                buffer_idx += item.value_size;
                break;
            case bson_value_type::bson_string:
                // write string size
                tmp_ptr = reinterpret_cast<uint32_t *>(buffer_ + buffer_idx);
                tmp_ptr[0] = item.value_size;
                buffer_idx += sizeof(uint32_t);

                // write string data
                std::memcpy(buffer_ + buffer_idx, item.value, item.value_size);
                buffer_idx += item.value_size;
                break;
            case bson_value_type::bson_binary:
                // write binary size
                tmp_ptr = reinterpret_cast<uint32_t *>(buffer_ + buffer_idx);
                tmp_ptr[0] = item.value_size;
                buffer_idx += sizeof(uint32_t);

                // write binary subtype
                buffer_[buffer_idx] = (unsigned char)bson_value_type::bson_binary_subtype;
                buffer_idx += sizeof(char);

                // write binary data
                // std::cout << "bson binary at " << buffer_idx << std::endl;

                // printf("bson size %d for %s binary at %d for bytes %d\n", size_, item.key.c_str(), buffer_idx, item.value_size);
                std::memcpy(buffer_ + buffer_idx, item.value, item.value_size);
                // printf("memcpy'ed %d bytes\n", item.value_size);
                buffer_idx += item.value_size;
                break;
            default:
                break;
            }
            // std::stringstream ss;
            // for (int i = 0; i < buffer_idx; ++i)
            //     ss << std::hex << (int)buffer_[i] << ",";
            // std::string mystr = ss.str();
            // std::cout << mystr << std::endl;

            // std::memcpy(buffer_.get() + buffer_idx, item.value, item.value_size);
            // std::cout << "bson processed " << item.key
            //           << " " << item.value_size
            //           << std::endl;
            // buffer_idx += item.value_size;
        }

        // printf("bson trailing 0 at %d\n", buffer_idx);
        buffer_[buffer_idx] = 0; // trailing 0x00
    }

    uint8_t *Bson::ptr()
    {
        return buffer_;
    }

    // std::shared_ptr<unsigned char[]> Bson::get_buffer()
    // {
    //     return buffer_;
    // }

    uint32_t Bson::size()
    {
        return size_;
    }

} // namespace Jetraser