
#include "Context.h"
#include <iostream>
#include <signal.h>
#include <unistd.h> // for sleep function
#include "MainEventsLoop.h"
#include "Events/BaseEvent.h"

using namespace std;

static bool quit = false;

static void signal_handle(int signum)
{
    printf("Quit due to CTRL+C command from user!\n");
    quit = true;
}

int main(int argc, char *argv[])
{
    Jetracer::context_t ctx;
    ctx.stream_video = new Jetracer::Ordered<bool>(false); // do not stream video by default

    /* Register a shuwdown handler to ensure
       a clean shutdown if user types <ctrl+c> */
    struct sigaction sig_action;
    sig_action.sa_handler = signal_handle;
    sigemptyset(&sig_action.sa_mask);
    sig_action.sa_flags = 0;
    sigaction(SIGINT, &sig_action, NULL);

    // start main loop
    cout << "create mainLoop" << endl;
    Jetracer::MainEventsLoop mainLoop("MainLoop", &ctx);
    mainLoop.createThread();

    cout << "entering infinite mainLoop" << endl;
    while (!quit)
    {
        sleep(1);
        // cout << "quit=" << quit << endl;
    };

    // caught CTRL+C
    cout << "caught CTRL+C" << endl;
    cout << "Closing thread: mainLoop in main" << endl;
    Jetracer::pEvent event = std::make_shared<Jetracer::BaseEvent>();
    event->event_type = Jetracer::EventType::event_stop_thread;
    mainLoop.pushEvent(event);
    cout << "Main sleep" << endl;
    // sleep(5);
    mainLoop.exitThread();
    return 0;
}
