/* SPDX-License-Identifier: BSD-2-Clause */

/*
 * The SIP module interface.
 *
 * Copyright (c) 2025 Phil Thompson <phil@riverbankcomputing.com>
 */


#ifndef _SIP_H
#define _SIP_H


#include <Python.h>

/* Sanity check on the Python version. */
#if PY_VERSION_HEX < 0x03090000
#error "This version of PyQt5.sip requires Python v3.9 or later"
#endif


#ifdef __cplusplus
#include <exception>

typedef bool (*sipExceptionHandler)(std::exception_ptr);
#else
typedef void *sipExceptionHandler;
#endif


#ifdef __cplusplus
extern "C" {
#endif


/* The version of the ABI. */
#define SIP_ABI_MAJOR_VERSION       13
#define SIP_ABI_MINOR_VERSION       10
#define SIP_MODULE_PATCH_VERSION    0


/*
 * The change history of the ABI.
 *
 * v13.10
 *  - Added SIP_ABI_VERSION as a module attribute.
 *
 * v13.9
 *  - Python v3.9 or later is required.
 *  - Added a new implementation of sipDeprecated() that takes an optional
 *    supplementary message.
 *
 * v13.8
 *  - Added the 'I' conversion character to the argument and result parsers.
 *
 * v13.7
 *  - Added support for Python v3.13.
 *  - Python v3.8 or later is required.
 *  - C99 support is assumed.
 *
 * v13.6
 *  - Added support for Python v3.12.
 *  - Added sipPyTypeDictRef().
 *  - Deprecated sipPyTypeDict().
 *
 * v13.5
 *  - Added the '#' conversion character to the argument parsers.
 *
 * v13.4
 *  - The enum implementation now supports missing members.
 *  - Published the 'array' type.
 *  - Added the ctd_sizeof, ctd_array_delete members to sipClassTypeDef.
 *  - Added the '>' conversion character to the argument parsers.
 *
 * v13.3
 *  - Python v3.7 or later is required.
 *
 * v13.2
 *  - Added support for SIP_ENUM_UINT_ENUM to represent IntEnums with unsigned
 *    members.
 *
 * v13.1
 *  - Added sipNextExceptionHandler().
 */


/* The version of the code generator. */
#define SIP_VERSION                 0x60a00
#define SIP_VERSION_STR             "6.10.0"

/* These are all dependent on the user-specified name of the sip module. */
#define _SIP_MODULE_FQ_NAME         "PyQt5.sip"
#define _SIP_MODULE_NAME            "sip"
#define _SIP_MODULE_SHARED          1
#define _SIP_MODULE_ENTRY           PyInit_sip


/*
 * Qt includes this typedef and its meta-object system explicitly converts
 * types to uint.  If these correspond to signal arguments then that conversion
 * is exposed.  Therefore SIP generates code that uses it.  This definition is
 * for the cases that SIP is generating non-Qt related bindings with compilers
 * that don't include it themselves (i.e. MSVC).
 */
typedef unsigned int uint;


/* Some C++ compatibility stuff. */
#if defined(__cplusplus)

/*
 * Cast a PyCFunctionWithKeywords to a PyCFunction in such a way that it
 * suppresses the GCC -Wcast-function-type warning.
 */
#define SIP_MLMETH_CAST(m)  reinterpret_cast<PyCFunction>(reinterpret_cast<void (*)(void)>(m))

#if __cplusplus >= 201103L || defined(_MSVC_LANG)

/* C++11 and later. */
#define SIP_NULLPTR     nullptr
#define SIP_OVERRIDE    override

#else

/* Earlier versions of C++. */
#define SIP_NULLPTR     NULL
#define SIP_OVERRIDE

#endif

#else

/* Cast a PyCFunctionWithKeywords to a PyCFunction. */
#define SIP_MLMETH_CAST(m)  ((PyCFunction)(m))

/* C. */
#define SIP_NULLPTR     NULL
#define SIP_OVERRIDE

#endif


/*
 * The mask that can be passed to sipTrace().
 */
#define SIP_TRACE_CATCHERS  0x0001
#define SIP_TRACE_CTORS     0x0002
#define SIP_TRACE_DTORS     0x0004
#define SIP_TRACE_INITS     0x0008
#define SIP_TRACE_DEALLOCS  0x0010
#define SIP_TRACE_METHODS   0x0020


/*
 * Hide some thread dependent stuff.  This can be removed when support for
 * Python v3.6 is removed.
 */
#ifdef WITH_THREAD
typedef PyGILState_STATE sip_gilstate_t;
#define SIP_RELEASE_GIL(gs) PyGILState_Release(gs);
#define SIP_BLOCK_THREADS   {PyGILState_STATE sipGIL = PyGILState_Ensure();
#define SIP_UNBLOCK_THREADS PyGILState_Release(sipGIL);}
#else
typedef int sip_gilstate_t;
#define SIP_RELEASE_GIL(gs)
#define SIP_BLOCK_THREADS
#define SIP_UNBLOCK_THREADS
#endif


/*
 * Forward declarations of types.
 */
struct _sipBufferDef;
typedef struct _sipBufferDef sipBufferDef;

struct _sipBufferInfoDef;
typedef struct _sipBufferInfoDef sipBufferInfoDef;

struct _sipCFunctionDef;
typedef struct _sipCFunctionDef sipCFunctionDef;

struct _sipDateDef;
typedef struct _sipDateDef sipDateDef;

struct _sipMethodDef;
typedef struct _sipMethodDef sipMethodDef;

struct _sipSimpleWrapper;
typedef struct _sipSimpleWrapper sipSimpleWrapper;

struct _sipTimeDef;
typedef struct _sipTimeDef sipTimeDef;

struct _sipTypeDef;
typedef struct _sipTypeDef sipTypeDef;

struct _sipWrapperType;
typedef struct _sipWrapperType sipWrapperType;

struct _sipWrapper;
typedef struct _sipWrapper sipWrapper;


/*
 * The different events a handler can be registered for.
 */
typedef enum
{
    sipEventWrappedInstance,    /* After wrapping a C/C++ instance. */
    sipEventCollectingWrapper,  /* When garbage collecting a wrapper object. */
    sipEventNrEvents
} sipEventType;

/*
 * The event handlers.
 */
typedef void (*sipWrappedInstanceEventHandler)(void *sipCpp);
typedef void (*sipCollectingWrapperEventHandler)(sipSimpleWrapper *sipSelf);


/*
 * The operation an access function is being asked to perform.
 */
typedef enum
{
    UnguardedPointer,   /* Return the unguarded pointer. */
    GuardedPointer,     /* Return the guarded pointer, ie. 0 if it has gone. */
    ReleaseGuard        /* Release the guard, if any. */
} AccessFuncOp;


/*
 * Some convenient function pointers.
 */
typedef void *(*sipInitFunc)(sipSimpleWrapper *, PyObject *, PyObject *,
        PyObject **, PyObject **, PyObject **);
typedef int (*sipFinalFunc)(PyObject *, void *, PyObject *, PyObject **);
typedef void *(*sipAccessFunc)(sipSimpleWrapper *, AccessFuncOp);
typedef int (*sipTraverseFunc)(void *, visitproc, void *);
typedef int (*sipClearFunc)(void *);
typedef int (*sipGetBufferFuncLimited)(PyObject *, void *, sipBufferDef *);
typedef void (*sipReleaseBufferFuncLimited)(PyObject *, void *);
#if !defined(Py_LIMITED_API)
typedef int (*sipGetBufferFunc)(PyObject *, void *, Py_buffer *, int);
typedef void (*sipReleaseBufferFunc)(PyObject *, void *, Py_buffer *);
#endif
typedef void (*sipDeallocFunc)(sipSimpleWrapper *);
typedef void *(*sipCastFunc)(void *, const sipTypeDef *);
typedef const sipTypeDef *(*sipSubClassConvertFunc)(void **);
typedef int (*sipConvertToFunc)(PyObject *, void **, int *, PyObject *,
        void **);
typedef PyObject *(*sipConvertFromFunc)(void *, PyObject *);
typedef void (*sipVirtErrorHandlerFunc)(sipSimpleWrapper *, sip_gilstate_t);
typedef int (*sipVirtHandlerFunc)(sip_gilstate_t, sipVirtErrorHandlerFunc,
        sipSimpleWrapper *, PyObject *, ...);
typedef void (*sipAssignFunc)(void *, Py_ssize_t, void *);
typedef void *(*sipArrayFunc)(Py_ssize_t);
typedef void (*sipArrayDeleteFunc)(void *);
typedef void *(*sipCopyFunc)(const void *, Py_ssize_t);
typedef void (*sipReleaseFunc)(void *, int);
typedef void (*sipReleaseUSFunc)(void *, int, void *);
typedef PyObject *(*sipPickleFunc)(void *);
typedef int (*sipAttrGetterFunc)(const sipTypeDef *, PyObject *);
typedef PyObject *(*sipVariableGetterFunc)(void *, PyObject *, PyObject *);
typedef int (*sipVariableSetterFunc)(void *, PyObject *, PyObject *);
typedef void *(*sipProxyResolverFunc)(void *);
typedef void (*sipWrapperVisitorFunc)(sipSimpleWrapper *, void *);


#if !defined(Py_LIMITED_API)
/*
 * The meta-type of a wrapper type.
 */
struct _sipWrapperType {
    /*
     * The super-metatype.  This must be first in the structure so that it can
     * be cast to a PyTypeObject *.
     */
    PyHeapTypeObject super;

    /* Set if the type is a user implemented Python sub-class. */
    unsigned wt_user_type : 1;

    /* Set if the type's dictionary contains all lazy attributes. */
    unsigned wt_dict_complete : 1;

    /* Unused and available for future use. */
    unsigned wt_unused : 30;

    /* The generated type structure. */
    sipTypeDef *wt_td;

    /* The list of init extenders. */
    struct _sipInitExtenderDef *wt_iextend;

    /*
     * For the user to use.  Note that any data structure will leak if the
     * type is garbage collected.
     */
    void *wt_user_data;

    /* Reserved for future use by this ABI version. */
    void *wt_reserved; 
};


/*
 * The type of a simple C/C++ wrapper object.
 */
struct _sipSimpleWrapper {
    PyObject_HEAD

    /*
     * The data, initially a pointer to the C/C++ object, as interpreted by the
     * access function.
     */
    void *data;

    /* The optional access function. */
    sipAccessFunc access_func;

    /* Object flags. */
    unsigned sw_flags;

    /* The optional dictionary of extra references keyed by argument number. */
    PyObject *extra_refs;

    /* For the user to use. */
    PyObject *user;

    /* The instance dictionary. */
    PyObject *dict;

    /* The main instance if this is a mixin. */
    PyObject *mixin_main;

    /* Reserved for future use by this ABI version. */
    void *reserved; 

    /* Next object at this address. */
    struct _sipSimpleWrapper *next;
};


/*
 * The type of a C/C++ wrapper object that supports parent/child relationships.
 */
struct _sipWrapper {
    /* The super-type. */
    sipSimpleWrapper super;

    /* First child object. */
    struct _sipWrapper *first_child;

    /* Next sibling. */
    struct _sipWrapper *sibling_next;

    /* Previous sibling. */
    struct _sipWrapper *sibling_prev;

    /* Owning object. */
    struct _sipWrapper *parent;
};
#endif


/*
 * The information describing an encoded type ID.
 */
typedef struct _sipEncodedTypeDef {
    /* The type number. */
    unsigned sc_type : 16;

    /* The module number (255 for this one). */
    unsigned sc_module : 8;

    /* A context specific flag. */
    unsigned sc_flag : 1;
} sipEncodedTypeDef;


/*
 * The information describing static instances.
 */
typedef struct _sipInstancesDef {
    /* The types. */
    struct _sipTypeInstanceDef *id_type;

    /* The void *. */
    struct _sipVoidPtrInstanceDef *id_voidp;

    /* The chars. */
    struct _sipCharInstanceDef *id_char;

    /* The strings. */
    struct _sipStringInstanceDef *id_string;

    /* The ints. */
    struct _sipIntInstanceDef *id_int;

    /* The longs. */
    struct _sipLongInstanceDef *id_long;

    /* The unsigned longs. */
    struct _sipUnsignedLongInstanceDef *id_ulong;

    /* The long longs. */
    struct _sipLongLongInstanceDef *id_llong;

    /* The unsigned long longs. */
    struct _sipUnsignedLongLongInstanceDef *id_ullong;

    /* The doubles. */
    struct _sipDoubleInstanceDef *id_double;
} sipInstancesDef;


/*
 * The information describing a type initialiser extender.
 */
typedef struct _sipInitExtenderDef {
    /* The extender function. */
    sipInitFunc ie_extender;

    /* The class being extended. */
    sipEncodedTypeDef ie_class;

    /* The next extender for this class. */
    struct _sipInitExtenderDef *ie_next;
} sipInitExtenderDef;


/*
 * The information describing a sub-class convertor.
 */
typedef struct _sipSubClassConvertorDef {
    /* The convertor. */
    sipSubClassConvertFunc scc_convertor;

    /* The encoded base type. */
    sipEncodedTypeDef scc_base;

    /* The base type. */
    struct _sipTypeDef *scc_basetype;
} sipSubClassConvertorDef;


/*
 * The structure populated by %BIGetBufferCode when the limited API is enabled.
 */
struct _sipBufferDef {
    /* The address of the buffer. */
    void *bd_buffer;

    /* The length of the buffer. */
    Py_ssize_t bd_length;

    /* Set if the buffer is read-only. */
    int bd_readonly;
};


/*
 * The structure describing a Python buffer.
 */
struct _sipBufferInfoDef {
    /* This is internal to sip. */
    void *bi_internal;

    /* The address of the buffer. */
    void *bi_buf;

    /* A reference to the object implementing the buffer interface. */
    PyObject *bi_obj;

    /* The length of the buffer in bytes. */
    Py_ssize_t bi_len;

    /* Set if the buffer is read-only. */
    int bi_readonly;

    /* The format of each element of the buffer. */
    char *bi_format;
};


/*
 * The structure describing a Python C function.
 */
struct _sipCFunctionDef {
    /* The C function. */
    PyMethodDef *cf_function;

    /* The optional bound object. */
    PyObject *cf_self;
};


/*
 * The structure describing a Python method.
 */
struct _sipMethodDef {
    /* The function that implements the method. */
    PyObject *pm_function;

    /* The bound object. */
    PyObject *pm_self;
};


/*
 * The structure describing a Python date.
 */
struct _sipDateDef {
    /* The year. */
    int pd_year;

    /* The month (1-12). */
    int pd_month;

    /* The day (1-31). */
    int pd_day;
};


/*
 * The structure describing a Python time.
 */
struct _sipTimeDef {
    /* The hour (0-23). */
    int pt_hour;

    /* The minute (0-59). */
    int pt_minute;

    /* The second (0-59). */
    int pt_second;

    /* The microsecond (0-999999). */
    int pt_microsecond;
};


/*
 * The different error states of handwritten code.
 */
typedef enum {
    sipErrorNone,       /* There is no error. */
    sipErrorFail,       /* The error is a failure. */
    sipErrorContinue    /* It may not apply if a later operation succeeds. */
} sipErrorState;


/*
 * The different Python slot types.  New slots must be added to the end,
 * otherwise the major version of the internal ABI must be changed.
 */
typedef enum {
    str_slot,           /* __str__ */
    int_slot,           /* __int__ */
    float_slot,         /* __float__ */
    len_slot,           /* __len__ */
    contains_slot,      /* __contains__ */
    add_slot,           /* __add__ for number */
    concat_slot,        /* __add__ for sequence types */
    sub_slot,           /* __sub__ */
    mul_slot,           /* __mul__ for number types */
    repeat_slot,        /* __mul__ for sequence types */
    div_slot,           /* __div__ */
    mod_slot,           /* __mod__ */
    floordiv_slot,      /* __floordiv__ */
    truediv_slot,       /* __truediv__ */
    and_slot,           /* __and__ */
    or_slot,            /* __or__ */
    xor_slot,           /* __xor__ */
    lshift_slot,        /* __lshift__ */
    rshift_slot,        /* __rshift__ */
    iadd_slot,          /* __iadd__ for number types */
    iconcat_slot,       /* __iadd__ for sequence types */
    isub_slot,          /* __isub__ */
    imul_slot,          /* __imul__ for number types */
    irepeat_slot,       /* __imul__ for sequence types */
    idiv_slot,          /* __idiv__ */
    imod_slot,          /* __imod__ */
    ifloordiv_slot,     /* __ifloordiv__ */
    itruediv_slot,      /* __itruediv__ */
    iand_slot,          /* __iand__ */
    ior_slot,           /* __ior__ */
    ixor_slot,          /* __ixor__ */
    ilshift_slot,       /* __ilshift__ */
    irshift_slot,       /* __irshift__ */
    invert_slot,        /* __invert__ */
    call_slot,          /* __call__ */
    getitem_slot,       /* __getitem__ */
    setitem_slot,       /* __setitem__ */
    delitem_slot,       /* __delitem__ */
    lt_slot,            /* __lt__ */
    le_slot,            /* __le__ */
    eq_slot,            /* __eq__ */
    ne_slot,            /* __ne__ */
    gt_slot,            /* __gt__ */
    ge_slot,            /* __ge__ */
    bool_slot,          /* __bool__, __nonzero__ */
    neg_slot,           /* __neg__ */
    repr_slot,          /* __repr__ */
    hash_slot,          /* __hash__ */
    pos_slot,           /* __pos__ */
    abs_slot,           /* __abs__ */
    index_slot,         /* __index__ */
    iter_slot,          /* __iter__ */
    next_slot,          /* __next__ */
    setattr_slot,       /* __setattr__, __delattr__ */
    matmul_slot,        /* __matmul__ (for Python v3.5 and later) */
    imatmul_slot,       /* __imatmul__ (for Python v3.5 and later) */
    await_slot,         /* __await__ (for Python v3.5 and later) */
    aiter_slot,         /* __aiter__ (for Python v3.5 and later) */
    anext_slot,         /* __anext__ (for Python v3.5 and later) */
} sipPySlotType;


/*
 * The information describing a Python slot function.
 */
typedef struct _sipPySlotDef {
    /* The function. */
    void *psd_func;

    /* The type. */
    sipPySlotType psd_type;
} sipPySlotDef;


/*
 * The information describing a Python slot extender.
 */
typedef struct _sipPySlotExtenderDef {
    /* The function. */
    void *pse_func;

    /* The type. */
    sipPySlotType pse_type;

    /* The encoded class. */
    sipEncodedTypeDef pse_class;
} sipPySlotExtenderDef;


/*
 * The information describing a typedef.
 */
typedef struct _sipTypedefDef {
    /* The typedef name. */
    const char *tdd_name;

    /* The typedef value. */
    const char *tdd_type_name;
} sipTypedefDef;


/*
 * The information describing a variable or property.
 */

typedef enum
{
    PropertyVariable,       /* A property. */
    InstanceVariable,       /* An instance variable. */
    ClassVariable           /* A class (i.e. static) variable. */
} sipVariableType;

typedef struct _sipVariableDef {
    /* The type of variable. */
    sipVariableType vd_type;

    /* The name. */
    const char *vd_name;

    /*
     * The getter.  If this is a variable (rather than a property) then the
     * actual type is sipVariableGetterFunc.
     */
    PyMethodDef *vd_getter;

    /*
     * The setter.  If this is a variable (rather than a property) then the
     * actual type is sipVariableSetterFunc.  It is NULL if the property cannot
     * be set or the variable is const.
     */
    PyMethodDef *vd_setter;

    /* The property deleter. */
    PyMethodDef *vd_deleter;

    /* The docstring. */
    const char *vd_docstring;
} sipVariableDef;


/*
 * The information describing a type, either a C++ class (or C struct), a C++
 * namespace, a mapped type or a named enum.
 */
struct _sipTypeDef {
    /*
     * The module, 0 if the type hasn't been initialised.
     */
    struct _sipExportedModuleDef *td_module;

    /* Type flags, see the sipType*() macros. */
    int td_flags;

    /* The C/C++ name of the type. */
    int td_cname;

    /* The Python type object. */
    PyTypeObject *td_py_type;

    /* Any additional fixed data generated by a plugin. */
    void *td_plugin_data;
};


/*
 * The information describing a container (ie. a class, namespace or a mapped
 * type).
 */
typedef struct _sipContainerDef {
    /*
     * The Python name of the type, -1 if this is a namespace extender (in the
     * context of a class) or doesn't require a namespace (in the context of a
     * mapped type). */
    int cod_name;

    /*
     * The scoping type or the namespace this is extending if it is a namespace
     * extender.
     */
    sipEncodedTypeDef cod_scope;

    /* The number of lazy methods. */
    int cod_nrmethods;

    /* The table of lazy methods. */
    PyMethodDef *cod_methods;

    /* The number of variables. */
    int cod_nrvariables;

    /* The table of variables. */
    sipVariableDef *cod_variables;

    /* The static instances. */
    sipInstancesDef cod_instances;
} sipContainerDef;


/*
 * The information describing a C++ class (or C struct) or a C++ namespace.
 */
typedef struct _sipClassTypeDef {
    /* The base type information. */
    sipTypeDef ctd_base;

    /* The container information. */
    sipContainerDef ctd_container;

    /* The docstring. */
    const char *ctd_docstring;

    /*
     * The meta-type name, -1 to use the meta-type of the first super-type
     * (normally sipWrapperType).
     */
    int ctd_metatype;

    /* The super-type name, -1 to use sipWrapper. */
    int ctd_supertype;

    /* The super-types. */
    sipEncodedTypeDef *ctd_supers;

    /* The table of Python slots. */
    sipPySlotDef *ctd_pyslots;

    /* The initialisation function. */
    sipInitFunc ctd_init;

    /* The traverse function. */
    sipTraverseFunc ctd_traverse;

    /* The clear function. */
    sipClearFunc ctd_clear;

    /* The get buffer function. */
#if defined(Py_LIMITED_API)
    sipGetBufferFuncLimited ctd_getbuffer;
#else
    sipGetBufferFunc ctd_getbuffer;
#endif

    /* The release buffer function. */
#if defined(Py_LIMITED_API)
    sipReleaseBufferFuncLimited ctd_releasebuffer;
#else
    sipReleaseBufferFunc ctd_releasebuffer;
#endif

    /* The deallocation function. */
    sipDeallocFunc ctd_dealloc;

    /* The optional assignment function. */
    sipAssignFunc ctd_assign;

    /* The optional array allocation function. */
    sipArrayFunc ctd_array;

    /* The optional copy function. */
    sipCopyFunc ctd_copy;

    /* The release function, 0 if a C struct. */
    sipReleaseFunc ctd_release;

    /* The cast function, 0 if a C struct. */
    sipCastFunc ctd_cast;

    /* The optional convert to function. */
    sipConvertToFunc ctd_cto;

    /* The optional convert from function. */
    sipConvertFromFunc ctd_cfrom;

    /* The next namespace extender. */
    struct _sipClassTypeDef *ctd_nsextender;

    /* The pickle function. */
    sipPickleFunc ctd_pickle;

    /* The finalisation function. */
    sipFinalFunc ctd_final;

    /* The mixin initialisation function. */
    initproc ctd_init_mixin;

    /* The optional array delete function. */
    sipArrayDeleteFunc ctd_array_delete;

    /* The sizeof the class. */
    size_t ctd_sizeof;
} sipClassTypeDef;


/*
 * The information describing a mapped type.
 */
typedef struct _sipMappedTypeDef {
    /* The base type information. */
    sipTypeDef mtd_base;

    /* The container information. */
    sipContainerDef mtd_container;

    /* The optional assignment function. */
    sipAssignFunc mtd_assign;

    /* The optional array allocation function. */
    sipArrayFunc mtd_array;

    /* The optional copy function. */
    sipCopyFunc mtd_copy;

    /* The optional release function. */
    sipReleaseUSFunc mtd_release;

    /* The optional convert to function. */
    sipConvertToFunc mtd_cto;

    /* The optional convert from function. */
    sipConvertFromFunc mtd_cfrom;
} sipMappedTypeDef;


/*
 * The information describing a named enum.
 */
typedef struct _sipEnumTypeDef {
    /* The base type information. */
    sipTypeDef etd_base;

    /* The Python base type. */
    int etd_base_type;

    /* The Python name of the enum. */
    int etd_name;

    /* The scoping type, -1 if it is defined at the module level. */
    int etd_scope;

    /* The number of members. */
    int etd_nr_members;

    /* The Python slots. */
    struct _sipPySlotDef *etd_pyslots;
} sipEnumTypeDef;


/*
 * The information describing an external type.
 */
typedef struct _sipExternalTypeDef {
    /* The index into the type table. */
    int et_nr;

    /* The name of the type. */
    const char *et_name;
} sipExternalTypeDef;


/*
 * Defines an entry in the module specific list of delayed dtor calls.
 */
typedef struct _sipDelayedDtor {
    /* The C/C++ instance. */
    void *dd_ptr;

    /* The class name. */
    const char *dd_name;

    /* Non-zero if dd_ptr is a derived class instance. */
    int dd_isderived;

    /* Next in the list. */
    struct _sipDelayedDtor *dd_next;
} sipDelayedDtor;


/*
 * Defines a virtual error handler.
 */
typedef struct _sipVirtErrorHandlerDef {
    /* The name of the handler. */
    const char *veh_name;

    /* The handler function. */
    sipVirtErrorHandlerFunc veh_handler;
} sipVirtErrorHandlerDef;


/*
 * Defines a type imported from another module.
 */
typedef union _sipImportedTypeDef {
    /* The type name before the module is imported. */
    const char *it_name;

    /* The type after the module is imported. */
    sipTypeDef *it_td;
} sipImportedTypeDef;


/*
 * Defines a virtual error handler imported from another module.
 */
typedef union _sipImportedVirtErrorHandlerDef {
    /* The handler name before the module is imported. */
    const char *iveh_name;

    /* The handler after the module is imported. */
    sipVirtErrorHandlerFunc iveh_handler;
} sipImportedVirtErrorHandlerDef;


/*
 * Defines an exception imported from another module.
 */
typedef union _sipImportedExceptionDef {
    /* The exception name before the module is imported. */
    const char *iexc_name;

    /* The exception object after the module is imported. */
    PyObject *iexc_object;
} sipImportedExceptionDef;


/*
 * The information describing an imported module.
 */
typedef struct _sipImportedModuleDef {
    /* The module name. */
    const char *im_name;

    /* The types imported from the module. */
    sipImportedTypeDef *im_imported_types;

    /* The virtual error handlers imported from the module. */
    sipImportedVirtErrorHandlerDef *im_imported_veh;

    /* The exceptions imported from the module. */
    sipImportedExceptionDef *im_imported_exceptions;
} sipImportedModuleDef;


/*
 * The main client module structure.
 */
typedef struct _sipExportedModuleDef {
    /* The next in the list. */
    struct _sipExportedModuleDef *em_next;

    /* The SIP API minor version number. */
    unsigned em_api_minor;

    /* The module name. */
    int em_name;

    /* The module name as an object. */
    PyObject *em_nameobj;

    /* The string pool. */
    const char *em_strings;

    /* The imported modules. */
    sipImportedModuleDef *em_imports;

    /* The number of types. */
    int em_nrtypes;

    /* The table of types. */
    sipTypeDef **em_types;

    /* The table of external types. */
    sipExternalTypeDef *em_external;

    /* The number of typedefs. */
    int em_nrtypedefs;

    /* The table of typedefs. */
    sipTypedefDef *em_typedefs;

    /* The table of virtual error handlers. */
    sipVirtErrorHandlerDef *em_virterrorhandlers;

    /* The sub-class convertors. */
    sipSubClassConvertorDef *em_convertors;

    /* The static instances. */
    sipInstancesDef em_instances;

    /* The license. */
    struct _sipLicenseDef *em_license;

    /* The table of exception types. */
    PyObject **em_exceptions;

    /* The table of Python slot extenders. */
    sipPySlotExtenderDef *em_slotextend;

    /* The table of initialiser extenders. */
    sipInitExtenderDef *em_initextend;

    /* The delayed dtor handler. */
    void (*em_delayeddtors)(const sipDelayedDtor *);

    /* The list of delayed dtors. */
    sipDelayedDtor *em_ddlist;

    /* The exception handler. */
    sipExceptionHandler em_exception_handler;
} sipExportedModuleDef;


/*
 * The information describing a license to be added to a dictionary.
 */
typedef struct _sipLicenseDef {
    /* The type of license. */
    const char *lc_type;

    /* The licensee. */
    const char *lc_licensee;

    /* The timestamp. */
    const char *lc_timestamp;

    /* The signature. */
    const char *lc_signature;
} sipLicenseDef;


/*
 * The information describing a void pointer instance to be added to a
 * dictionary.
 */
typedef struct _sipVoidPtrInstanceDef {
    /* The void pointer name. */
    const char *vi_name;

    /* The void pointer value. */
    void *vi_val;
} sipVoidPtrInstanceDef;


/*
 * The information describing a char instance to be added to a dictionary.
 */
typedef struct _sipCharInstanceDef {
    /* The char name. */
    const char *ci_name;

    /* The char value. */
    char ci_val;

    /* The encoding used, either 'A', 'L', '8' or 'N'. */
    char ci_encoding;
} sipCharInstanceDef;


/*
 * The information describing a string instance to be added to a dictionary.
 * This is also used as a hack to add (or fix) other types rather than add a
 * new table type and so requiring a new major version of the API.
 */
typedef struct _sipStringInstanceDef {
    /* The string name. */
    const char *si_name;

    /* The string value. */
    const char *si_val;

    /*
     * The encoding used, either 'A', 'L', '8' or 'N'.  'w' and 'W' are also
     * used to support the fix for wchar_t.
     */
    char si_encoding;
} sipStringInstanceDef;


/*
 * The information describing an int instance to be added to a dictionary.
 */
typedef struct _sipIntInstanceDef {
    /* The int name. */
    const char *ii_name;

    /* The int value. */
    int ii_val;
} sipIntInstanceDef;


/*
 * The information describing a long instance to be added to a dictionary.
 */
typedef struct _sipLongInstanceDef {
    /* The long name. */
    const char *li_name;

    /* The long value. */
    long li_val;
} sipLongInstanceDef;


/*
 * The information describing an unsigned long instance to be added to a
 * dictionary.
 */
typedef struct _sipUnsignedLongInstanceDef {
    /* The unsigned long name. */
    const char *uli_name;

    /* The unsigned long value. */
    unsigned long uli_val;
} sipUnsignedLongInstanceDef;


/*
 * The information describing a long long instance to be added to a dictionary.
 */
typedef struct _sipLongLongInstanceDef {
    /* The long long name. */
    const char *lli_name;

    /* The long long value. */
    long long lli_val;
} sipLongLongInstanceDef;


/*
 * The information describing an unsigned long long instance to be added to a
 * dictionary.
 */
typedef struct _sipUnsignedLongLongInstanceDef {
    /* The unsigned long long name. */
    const char *ulli_name;

    /* The unsigned long long value. */
    unsigned long long ulli_val;
} sipUnsignedLongLongInstanceDef;


/*
 * The information describing a double instance to be added to a dictionary.
 */
typedef struct _sipDoubleInstanceDef {
    /* The double name. */
    const char *di_name;

    /* The double value. */
    double di_val;
} sipDoubleInstanceDef;


/*
 * The information describing a class or enum instance to be added to a
 * dictionary.
 */
typedef struct _sipTypeInstanceDef {
    /* The type instance name. */
    const char *ti_name;

    /* The actual instance. */
    void *ti_ptr;

    /* A pointer to the generated type. */
    struct _sipTypeDef **ti_type;

    /* The wrapping flags. */
    int ti_flags;
} sipTypeInstanceDef;


/*
 * The API exported by the SIP module, ie. pointers to all the data and
 * functions that can be used by generated code.
 */
typedef struct _sipAPIDef {
    /*
     * This must be the first entry and it's signature must not change so that
     * version number mismatches can be detected and reported.
     */
    int (*api_export_module)(sipExportedModuleDef *client, unsigned api_major,
            unsigned api_minor, void *unused);

    /*
     * The following are part of the public API.
     */
    PyTypeObject *api_simplewrapper_type;
    PyTypeObject *api_wrapper_type;
    PyTypeObject *api_wrappertype_type;
    PyTypeObject *api_voidptr_type;

    void (*api_bad_catcher_result)(PyObject *method);
    void (*api_bad_length_for_slice)(Py_ssize_t seqlen, Py_ssize_t slicelen);
    PyObject *(*api_build_result)(int *isErr, const char *fmt, ...);
    PyObject *(*api_call_method)(int *isErr, PyObject *method, const char *fmt,
            ...);
    void (*api_call_procedure_method)(sip_gilstate_t, sipVirtErrorHandlerFunc,
            sipSimpleWrapper *, PyObject *, const char *, ...);
    Py_ssize_t (*api_convert_from_sequence_index)(Py_ssize_t idx,
            Py_ssize_t len);
    int (*api_can_convert_to_type)(PyObject *pyObj, const sipTypeDef *td,
            int flags);
    void *(*api_convert_to_type)(PyObject *pyObj, const sipTypeDef *td,
            PyObject *transferObj, int flags, int *statep, int *iserrp);
    void *(*api_convert_to_type_us)(PyObject *pyObj, const sipTypeDef *td,
            PyObject *transferObj, int flags, int *statep, void **user_statep,
            int *iserrp);
    void *(*api_force_convert_to_type)(PyObject *pyObj, const sipTypeDef *td,
            PyObject *transferObj, int flags, int *statep, int *iserrp);
    void *(*api_force_convert_to_type_us)(PyObject *pyObj,
            const sipTypeDef *td, PyObject *transferObj, int flags,
            int *statep, void **user_statep, int *iserrp);
    void (*api_release_type)(void *cpp, const sipTypeDef *td, int state);
    void (*api_release_type_us)(void *cpp, const sipTypeDef *td, int state,
            void *user_state);
    PyObject *(*api_convert_from_type)(void *cpp, const sipTypeDef *td,
            PyObject *transferObj);
    PyObject *(*api_convert_from_new_type)(void *cpp, const sipTypeDef *td,
            PyObject *transferObj);
    PyObject *(*api_convert_from_enum)(int eval, const sipTypeDef *td);
    int (*api_get_state)(PyObject *transferObj);
    void (*api_free)(void *mem);
    PyObject *(*api_get_pyobject)(void *cppPtr, const sipTypeDef *td);
    void *(*api_malloc)(size_t nbytes);
    int (*api_parse_result)(int *isErr, PyObject *method, PyObject *res,
            const char *fmt, ...);
    void (*api_trace)(unsigned mask, const char *fmt, ...);
    void (*api_transfer_back)(PyObject *self);
    void (*api_transfer_to)(PyObject *self, PyObject *owner);
    unsigned long (*api_long_as_unsigned_long)(PyObject *o);
    PyObject *(*api_convert_from_void_ptr)(void *val);
    PyObject *(*api_convert_from_const_void_ptr)(const void *val);
    PyObject *(*api_convert_from_void_ptr_and_size)(void *val,
            Py_ssize_t size);
    PyObject *(*api_convert_from_const_void_ptr_and_size)(const void *val,
            Py_ssize_t size);
    void *(*api_convert_to_void_ptr)(PyObject *obj);
    int (*api_export_symbol)(const char *name, void *sym);
    void *(*api_import_symbol)(const char *name);
    const sipTypeDef *(*api_find_type)(const char *type);
    int (*api_register_py_type)(PyTypeObject *type);
    const sipTypeDef *(*api_type_from_py_type_object)(PyTypeObject *py_type);
    const sipTypeDef *(*api_type_scope)(const sipTypeDef *td);
    const char *(*api_resolve_typedef)(const char *name);
    int (*api_register_attribute_getter)(const sipTypeDef *td,
            sipAttrGetterFunc getter);
    sipErrorState (*api_bad_callable_arg)(int arg_nr, PyObject *arg);
    void *(*api_get_address)(struct _sipSimpleWrapper *w);
    int (*api_enable_autoconversion)(const sipTypeDef *td, int enable);
    void *(*api_get_mixin_address)(struct _sipSimpleWrapper *w,
            const sipTypeDef *td);
    PyObject *(*api_convert_from_new_pytype)(void *cpp, PyTypeObject *py_type,
            sipWrapper *owner, sipSimpleWrapper **selfp, const char *fmt, ...);
    PyObject *(*api_convert_to_typed_array)(void *data, const sipTypeDef *td,
            const char *format, size_t stride, Py_ssize_t len, int flags);
    PyObject *(*api_convert_to_array)(void *data, const char *format,
            Py_ssize_t len, int flags);
    int (*api_register_proxy_resolver)(const sipTypeDef *td,
            sipProxyResolverFunc resolver);
    PyInterpreterState *(*api_get_interpreter)(void);
    void (*api_set_type_user_data)(sipWrapperType *, void *);
    void *(*api_get_type_user_data)(const sipWrapperType *);
    PyObject *(*api_py_type_dict)(const PyTypeObject *);
    const char *(*api_py_type_name)(const PyTypeObject *);
    int (*api_get_method)(PyObject *, sipMethodDef *);
    PyObject *(*api_from_method)(const sipMethodDef *);
    int (*api_get_c_function)(PyObject *, sipCFunctionDef *);
    int (*api_get_date)(PyObject *, sipDateDef *);
    PyObject *(*api_from_date)(const sipDateDef *);
    int (*api_get_datetime)(PyObject *, sipDateDef *, sipTimeDef *);
    PyObject *(*api_from_datetime)(const sipDateDef *, const sipTimeDef *);
    int (*api_get_time)(PyObject *, sipTimeDef *);
    PyObject *(*api_from_time)(const sipTimeDef *);
    int (*api_is_user_type)(const sipWrapperType *);
    int (*api_check_plugin_for_type)(const sipTypeDef *, const char *);
    PyObject *(*api_unicode_new)(Py_ssize_t, unsigned, int *, void **);
    void (*api_unicode_write)(int, void *, int, unsigned);
    void *(*api_unicode_data)(PyObject *, int *, Py_ssize_t *);
    int (*api_get_buffer_info)(PyObject *, sipBufferInfoDef *);
    void (*api_release_buffer_info)(sipBufferInfoDef *);
    PyObject *(*api_get_user_object)(const sipSimpleWrapper *);
    void (*api_set_user_object)(sipSimpleWrapper *, PyObject *);
    void (*api_instance_destroyed)(sipSimpleWrapper *sipSelf);
    int (*api_is_owned_by_python)(sipSimpleWrapper *);
    int (*api_enable_gc)(int enable);
    void (*api_print_object)(PyObject *o);
    int (*api_register_event_handler)(sipEventType type, const sipTypeDef *td,
            void *handler);
    int (*api_convert_to_enum)(PyObject *obj, const sipTypeDef *td);
    int (*api_convert_to_bool)(PyObject *obj);
    char (*api_long_as_char)(PyObject *o);
    signed char (*api_long_as_signed_char)(PyObject *o);
    unsigned char (*api_long_as_unsigned_char)(PyObject *o);
    short (*api_long_as_short)(PyObject *o);
    unsigned short (*api_long_as_unsigned_short)(PyObject *o);
    int (*api_long_as_int)(PyObject *o);
    unsigned int (*api_long_as_unsigned_int)(PyObject *o);
    long (*api_long_as_long)(PyObject *o);
    long long (*api_long_as_long_long)(PyObject *o);
    unsigned long long (*api_long_as_unsigned_long_long)(PyObject *o);
    int (*api_convert_from_slice_object)(PyObject *slice, Py_ssize_t length,
            Py_ssize_t *start, Py_ssize_t *stop, Py_ssize_t *step,
            Py_ssize_t *slicelength);
    size_t (*api_long_as_size_t)(PyObject *o);
    void (*api_visit_wrappers)(sipWrapperVisitorFunc visitor, void *closure);
    int (*api_register_exit_notifier)(PyMethodDef *md);
    int (*api_is_enum_flag)(PyObject *o);
    PyObject *(*api_py_type_dict_ref)(PyTypeObject *);
    void (*unused_public_2)(void);
    void (*unused_public_3)(void);
    void (*unused_public_4)(void);
    void (*unused_public_5)(void);

    /*
     * The following are not part of the public API.
     */
    int (*api_init_module)(sipExportedModuleDef *client, PyObject *mod_dict);
    int (*api_parse_args)(PyObject **parseErrp, PyObject *sipArgs,
            const char *fmt, ...);
    int (*api_parse_pair)(PyObject **parseErrp, PyObject *arg0, PyObject *arg1,
            const char *fmt, ...);
    void (*api_no_function)(PyObject *parseErr, const char *func,
            const char *doc);
    void (*api_no_method)(PyObject *parseErr, const char *scope,
            const char *method, const char *doc);
    void (*api_abstract_method)(const char *classname, const char *method);
    void (*api_bad_class)(const char *classname);
    void *(*api_get_cpp_ptr)(sipSimpleWrapper *w, const sipTypeDef *td);
    void *(*api_get_complex_cpp_ptr)(sipSimpleWrapper *w);
    PyObject *(*api_is_py_method)(sip_gilstate_t *gil, char *pymc,
            sipSimpleWrapper *sipSelf, const char *cname, const char *mname);
    void (*api_call_hook)(const char *hookname);
    void (*api_end_thread)(void);
    void (*api_raise_unknown_exception)(void);
    void (*api_raise_type_exception)(const sipTypeDef *td, void *ptr);
    int (*api_add_type_instance)(PyObject *dict, const char *name,
            void *cppPtr, const sipTypeDef *td);
    void (*api_bad_operator_arg)(PyObject *self, PyObject *arg,
            sipPySlotType st);
    PyObject *(*api_pyslot_extend)(sipExportedModuleDef *mod, sipPySlotType st,
            const sipTypeDef *type, PyObject *arg0, PyObject *arg1);
    void (*api_add_delayed_dtor)(sipSimpleWrapper *w);
    char (*api_bytes_as_char)(PyObject *obj);
    const char *(*api_bytes_as_string)(PyObject *obj);
    char (*api_string_as_ascii_char)(PyObject *obj);
    const char *(*api_string_as_ascii_string)(PyObject **obj);
    char (*api_string_as_latin1_char)(PyObject *obj);
    const char *(*api_string_as_latin1_string)(PyObject **obj);
    char (*api_string_as_utf8_char)(PyObject *obj);
    const char *(*api_string_as_utf8_string)(PyObject **obj);
#if defined(HAVE_WCHAR_H)
    wchar_t (*api_unicode_as_wchar)(PyObject *obj);
    wchar_t *(*api_unicode_as_wstring)(PyObject *obj);
#else
    int (*api_unicode_as_wchar)(PyObject *obj);
    int *(*api_unicode_as_wstring)(PyObject *obj);
#endif
    int (*api_deprecated)(const char *classname, const char *method);
    void (*api_keep_reference)(PyObject *self, int key, PyObject *obj);
    int (*api_parse_kwd_args)(PyObject **parseErrp, PyObject *sipArgs,
            PyObject *sipKwdArgs, const char **kwdlist, PyObject **unused,
            const char *fmt, ...);
    void (*api_add_exception)(sipErrorState es, PyObject **parseErrp);
    int (*api_parse_result_ex)(sip_gilstate_t, sipVirtErrorHandlerFunc,
            sipSimpleWrapper *, PyObject *method, PyObject *res,
            const char *fmt, ...);
    void (*api_call_error_handler)(sipVirtErrorHandlerFunc,
            sipSimpleWrapper *, sip_gilstate_t);
    int (*api_init_mixin)(PyObject *self, PyObject *args, PyObject *kwds,
            const sipClassTypeDef *ctd);
    PyObject *(*api_get_reference)(PyObject *self, int key);
    int (*api_is_derived_class)(sipSimpleWrapper *);
    void (*api_instance_destroyed_ex)(sipSimpleWrapper **sipSelfp);
    PyObject *(*api_is_py_method_12_8)(sip_gilstate_t *gil, char *pymc,
            sipSimpleWrapper **sipSelfp, const char *cname, const char *mname);
    sipExceptionHandler (*api_next_exception_handler)(void **statep);
    int (*api_deprecated_13_9)(const char *classname, const char *method, const char *message);
    void (*unused_private_3)(void);
    void (*unused_private_4)(void);
    void (*unused_private_5)(void);
} sipAPIDef;

const sipAPIDef *sip_init_library(PyObject *mod_dict);


/*
 * These are flags that can be passed to sipCanConvertToType(),
 * sipConvertToType() and sipForceConvertToType().
 */
#define SIP_NOT_NONE        0x01    /* Disallow None. */
#define SIP_NO_CONVERTORS   0x02    /* Disable any type convertors. */


/*
 * These are flags that can be passed to sipConvertToArray().
 */
#define SIP_READ_ONLY       0x01    /* The array is read-only. */
#define SIP_OWNS_MEMORY     0x02    /* The array owns its memory. */


/*
 * These are the state flags returned by %ConvertToTypeCode.  Note that the
 * values share the same "flagspace" as the contents of sw_flags.
 */
#define SIP_TEMPORARY       0x0001  /* A temporary instance. */
#define SIP_DERIVED_CLASS   0x0002  /* The instance is derived. */
#define SIP_USER            0x0004  /* For use by the user. */
#define SIP_RESERVED        0x0008  /* Reserved for future use by this ABI version. */


/*
 * These flags are specific to the Qt support API.
 */
#define SIP_SINGLE_SHOT     0x0001  /* The connection is single shot. */


/*
 * Useful macros, not part of the public API.
 */

/* These are held in sw_flags. */
#define SIP_INDIRECT        0x0010  /* If there is a level of indirection. */
#define SIP_ACCFUNC         0x0020  /* If there is an access function. */
#define SIP_NOT_IN_MAP      0x0040  /* If Python object is not in the map. */

#if !defined(Py_LIMITED_API)
#define SIP_PY_OWNED        0x0080  /* If owned by Python. */
#define SIP_SHARE_MAP       0x0100  /* If the map slot might be occupied. */
#define SIP_CPP_HAS_REF     0x0200  /* If C/C++ has a reference. */
#define SIP_POSSIBLE_PROXY  0x0400  /* If there might be a proxy slot. */
#define SIP_ALIAS           0x0800  /* If it is an alias. */
#define SIP_CREATED         0x1000  /* If the C/C++ object has been created. */

#define sipIsDerived(sw)    ((sw)->sw_flags & SIP_DERIVED_CLASS)
#define sipIsIndirect(sw)   ((sw)->sw_flags & SIP_INDIRECT)
#define sipIsAccessFunc(sw) ((sw)->sw_flags & SIP_ACCFUNC)
#define sipNotInMap(sw)     ((sw)->sw_flags & SIP_NOT_IN_MAP)
#define sipSetNotInMap(sw)  ((sw)->sw_flags |= SIP_NOT_IN_MAP)
#define sipIsPyOwned(sw)    ((sw)->sw_flags & SIP_PY_OWNED)
#define sipSetPyOwned(sw)   ((sw)->sw_flags |= SIP_PY_OWNED)
#define sipResetPyOwned(sw) ((sw)->sw_flags &= ~SIP_PY_OWNED)
#define sipCppHasRef(sw)    ((sw)->sw_flags & SIP_CPP_HAS_REF)
#define sipSetCppHasRef(sw) ((sw)->sw_flags |= SIP_CPP_HAS_REF)
#define sipResetCppHasRef(sw)   ((sw)->sw_flags &= ~SIP_CPP_HAS_REF)
#define sipPossibleProxy(sw)    ((sw)->sw_flags & SIP_POSSIBLE_PROXY)
#define sipSetPossibleProxy(sw) ((sw)->sw_flags |= SIP_POSSIBLE_PROXY)
#define sipIsAlias(sw)      ((sw)->sw_flags & SIP_ALIAS)
#define sipWasCreated(sw)   ((sw)->sw_flags & SIP_CREATED)
#endif

#define SIP_TYPE_TYPE_MASK  0x0003  /* The type type mask. */
#define SIP_TYPE_CLASS      0x0000  /* If the type is a C++ class. */
#define SIP_TYPE_NAMESPACE  0x0001  /* If the type is a C++ namespace. */
#define SIP_TYPE_MAPPED     0x0002  /* If the type is a mapped type. */
#define SIP_TYPE_ENUM       0x0003  /* If the type is an enum. */
#define SIP_TYPE_USER_STATE 0x0004  /* If the convertors need user state. */
#define SIP_TYPE_ABSTRACT   0x0008  /* If the type is abstract. */
#define SIP_TYPE_SCC        0x0010  /* If the type is subject to sub-class convertors. */
#define SIP_TYPE_ALLOW_NONE 0x0020  /* If the type can handle None. */
#define SIP_TYPE_STUB       0x0040  /* If the type is a stub. */
#define SIP_TYPE_NONLAZY    0x0080  /* If the type has a non-lazy method. */
#define SIP_TYPE_SUPER_INIT 0x0100  /* If the instance's super init should be called. */
#define SIP_TYPE_LIMITED_API    0x0200  /* Use the limited API.  If this is more generally required it may need to be moved to the module definition. */


/* The Python base types of enums. */
#define SIP_ENUM_ENUM       0       /* The base type is Enum. */
#define SIP_ENUM_FLAG       1       /* The base type is Flag. */
#define SIP_ENUM_INT_ENUM   2       /* The base type is IntEnum. */
#define SIP_ENUM_INT_FLAG   3       /* The base type is IntFlag. */
#define SIP_ENUM_UINT_ENUM  4       /* The base type is IntEnum with unsigned members. */


/*
 * The following are part of the public API.
 */
#define sipTypeIsClass(td)  (((td)->td_flags & SIP_TYPE_TYPE_MASK) == SIP_TYPE_CLASS)
#define sipTypeIsNamespace(td)  (((td)->td_flags & SIP_TYPE_TYPE_MASK) == SIP_TYPE_NAMESPACE)
#define sipTypeIsMapped(td) (((td)->td_flags & SIP_TYPE_TYPE_MASK) == SIP_TYPE_MAPPED)
#define sipTypeIsEnum(td)   (((td)->td_flags & SIP_TYPE_TYPE_MASK) == SIP_TYPE_ENUM)
#define sipTypeAsPyTypeObject(td)   ((td)->td_py_type)
#define sipTypeName(td)     sipNameFromPool((td)->td_module, (td)->td_cname)
#define sipTypePluginData(td)   ((td)->td_plugin_data)


/*
 * The following are not part of the public API.
 */
#define sipTypeNeedsUserState(td)   ((td)->td_flags & SIP_TYPE_USER_STATE)
#define sipTypeIsAbstract(td)   ((td)->td_flags & SIP_TYPE_ABSTRACT)
#define sipTypeHasSCC(td)   ((td)->td_flags & SIP_TYPE_SCC)
#define sipTypeAllowNone(td)    ((td)->td_flags & SIP_TYPE_ALLOW_NONE)
#define sipTypeIsStub(td)   ((td)->td_flags & SIP_TYPE_STUB)
#define sipTypeSetStub(td)  ((td)->td_flags |= SIP_TYPE_STUB)
#define sipTypeHasNonlazyMethod(td) ((td)->td_flags & SIP_TYPE_NONLAZY)
#define sipTypeCallSuperInit(td)    ((td)->td_flags & SIP_TYPE_SUPER_INIT)
#define sipTypeUseLimitedAPI(td)    ((td)->td_flags & SIP_TYPE_LIMITED_API)


/*
 * Get various names from the string pool for various data types.
 */
#define sipNameFromPool(em, mr) (&((em)->em_strings)[(mr)])
#define sipNameOfModule(em)     sipNameFromPool((em), (em)->em_name)
#define sipPyNameOfContainer(cod, td)   sipNameFromPool((td)->td_module, (cod)->cod_name)
#define sipPyNameOfEnum(etd)    sipNameFromPool((etd)->etd_base.td_module, (etd)->etd_name)


/*
 * The following are PyQt-specific extensions.  These may be removed in SIP v7.
 * Note that, at the moment, the PyQt5 and PyQt6 extensions are identical.
 */

/*
 * The description of a Qt signal for PyQt5.
 */
typedef int (*pyqt5EmitFunc)(void *, PyObject *);

typedef struct _pyqt5QtSignal {
    /* The normalised C++ name and signature of the signal. */
    const char *signature;

    /* The optional docstring. */
    const char *docstring;

    /*
     * If the signal is an overload of regular methods then this points to the
     * code that implements those methods.
     */
    PyMethodDef *non_signals;

    /*
     * If the signal has optional arguments then this function will implement
     * emit() for the signal.
     */
    pyqt5EmitFunc emitter;
} pyqt5QtSignal;


/*
 * This is the PyQt5-specific extension to the generated class type structure.
 */
typedef struct _pyqt5ClassPluginDef {
    /* A pointer to the QObject sub-class's staticMetaObject class variable. */
    const void *static_metaobject;

    /*
     * A set of flags.  At the moment only bit 0 is used to say if the type is
     * derived from QFlags.
     */
    unsigned flags;

    /*
     * The table of signals emitted by the type.  These are grouped by signal
     * name.
     */
    const pyqt5QtSignal *qt_signals;

    /* The name of the interface that the class defines. */
    const char *qt_interface;
} pyqt5ClassPluginDef;


/*
 * The description of a Qt signal for PyQt6.
 */
typedef int (*pyqt6EmitFunc)(void *, PyObject *);

typedef struct _pyqt6QtSignal {
    /* The normalised C++ name and signature of the signal. */
    const char *signature;

    /* The optional docstring. */
    const char *docstring;

    /*
     * If the signal is an overload of regular methods then this points to the
     * code that implements those methods.
     */
    PyMethodDef *non_signals;

    /*
     * If the signal has optional arguments then this function will implement
     * emit() for the signal.
     */
    pyqt6EmitFunc emitter;
} pyqt6QtSignal;


/*
 * This is the PyQt6-specific extension to the generated class type structure.
 */
typedef struct _pyqt6ClassPluginDef {
    /* A pointer to the QObject sub-class's staticMetaObject class variable. */
    const void *static_metaobject;

    /*
     * The table of signals emitted by the type.  These are grouped by signal
     * name.
     */
    const pyqt6QtSignal *qt_signals;

    /* The name of the interface that the class defines. */
    const char *qt_interface;
} pyqt6ClassPluginDef;


/*
 * This is the PyQt6-specific extension to the generated mapped type structure.
 */
typedef struct _pyqt6MappedTypePluginDef {
    /*
     * A set of flags.  At the moment only bit 0 is used to say if the type is
     * mapped to/from QFlags.
     */
    unsigned flags;
} pyqt6MappedTypePluginDef;


#ifdef __cplusplus
}
#endif


#endif
