#include "types.h"


#ifndef JETRACER_IMAGE_PIPELINE_THREAD_H
#define JETRACER_IMAGE_PIPELINE_THREAD_H


namespace Jetracer {

    class imagePipelineThread : public Thread {
    public:
        explicit imagePipelineThread(context_t * ctx)
                                    : _ctx(ctx)
        {
        }
        ~imagePipelineThread() {}
    private:
        virtual bool threadInitialize();
        virtual bool threadExecute();
        virtual bool threadShutdown();

        context_t * _ctx;
        int image_size;
    }
} // namespace Jetracer

#endif // JETRACER_IMAGE_PIPELINE_THREAD_H
