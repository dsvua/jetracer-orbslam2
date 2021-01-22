#include "Thread.h"

#ifndef JETRACER_ORB_EXTRACTOR_THREAD_H
#define JETRACER_ORB_EXTRACTOR_THREAD_H

namespace Jetracer {

    class orbExtractorThread : public Thread {
    public:
        explicit orbExtractorThread(int width, int height)
                                    : _width(width)
                                    , _height(height)
        {
        }
        ~orbExtractorThread() {}
    private:
        virtual bool threadInitialize();
        virtual bool threadExecute();
        virtual bool threadShutdown();

        int _width;
        int _height;

    }


} // namespace Jetracer

#endif // JETRACER_ORB_EXTRACTOR_THREAD_H

// currently not used

