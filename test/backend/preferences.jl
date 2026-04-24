using Preferences
using Suppressor
using ReTest

function test_preferences(bksym::Symbol)
    # Check current settings
    bkstr = lowercase(String(bksym))
    @test bkstr ∈ JACC.supported_backends
    @test JACC.backend == bkstr
    @test typeof(JACC._backend_dispatchable) == Val{Symbol(bkstr)}
    @test JACC.Preferences.Backend.default == bkstr
    @test JACC.Preferences.Backend._DEFAULT[] == bkstr
    @test bkstr ∈ JACC.Preferences.Backend.list
    @test bkstr ∈ JACC.Preferences.Backend._LIST[]
    if bkstr != "threads"
        @test JACC.Preferences.Backend._PLACE[][bkstr] == "weakdeps"
    else
        @test isempty(JACC.Preferences.Backend._PLACE[])
    end

    # Clear settings
    @suppress JACC.unset_backend()
    @test isempty(JACC.Preferences.Backend._LIST[])
    @test isempty(JACC.Preferences.Backend._PLACE[])
    @test load_preference(JACC, "backends") == nothing
    @test load_preference(JACC, "default_backend") == nothing

    # "not a backend"
    @test_throws ArgumentError JACC.set_backend("NAB")
    @test_throws ArgumentError JACC.set_default_backend("NAB")
    @test_throws ArgumentError JACC.add_backend("NAB")

    # Set again (and again...)
    @suppress JACC.set_backend(bkstr)
    @test load_preference(JACC, "backends") == [bkstr]
    @test load_preference(JACC, "default_backend") == bkstr
    @test JACC.Preferences.Backend._LIST[] == [bkstr]
    @test JACC.Preferences.Backend._DEFAULT[] == bkstr
    JACC.set_backend(bksym)
    @test load_preference(JACC, "backends") == [bkstr]
    @test load_preference(JACC, "default_backend") == bkstr
    @test JACC.Preferences.Backend._LIST[] == [bkstr]
    @test JACC.Preferences.Backend._DEFAULT[] == bkstr
    JACC.set_default_backend(bkstr)
    JACC.set_default_backend(bksym)
    JACC.add_backend(bkstr)
    JACC.add_backend(bksym)
    @test load_preference(JACC, "backends") == [bkstr]
    @test load_preference(JACC, "default_backend") == bkstr
    @test JACC.Preferences.Backend._LIST[] == [bkstr]
    @test JACC.Preferences.Backend._DEFAULT[] == bkstr

    # Use threads to test with multiple backends
    if bkstr != "threads"
        @suppress JACC.add_backend(:Threads)
        @test JACC.Preferences.Backend._LIST[] == [bkstr, "threads"]
        @test JACC.Preferences.Backend._DEFAULT[] == bkstr
        @suppress JACC.remove_backend(bkstr)
        @test haskey(JACC.proj().weakdeps, JACC.get_package_name(bkstr))
        @test JACC.Preferences.Backend._LIST[] == ["threads"]
        @test JACC.Preferences.Backend._DEFAULT[] == ""
        @test load_preference(JACC, "backends") == ["threads"]
        @test load_preference(JACC, "default_backend") == nothing
        @suppress JACC.add_backend(bkstr)
        @test JACC.Preferences.Backend._LIST[] == ["threads", bkstr]
        @test JACC.Preferences.Backend._DEFAULT[] == ""
        @test load_preference(JACC, "backends") == ["threads", bkstr]
        @test load_preference(JACC, "default_backend") == nothing
        @suppress JACC.set_default_backend("threads")
        @test JACC.Preferences.Backend._LIST[] == ["threads", bkstr]
        @test JACC.Preferences.Backend._DEFAULT[] == "threads"
        @test load_preference(JACC, "backends") == ["threads", bkstr]
        @test load_preference(JACC, "default_backend") == "threads"
        @suppress JACC.set_default_backend(bkstr)
        @test JACC.Preferences.Backend._LIST[] == ["threads", bkstr]
        @test JACC.Preferences.Backend._DEFAULT[] == bkstr
        @test load_preference(JACC, "backends") == ["threads", bkstr]
        @test load_preference(JACC, "default_backend") == bkstr
    end

    @suppress JACC.set_backend(bkstr)
end
