Java API
========

The Vortex Java API provides bindings for the Vortex library, enabling Java applications to work with Vortex arrays and files.

The API is split into two main components:

* **Vortex JNI**: Core JNI bindings for Vortex functionality
* **Vortex Spark**: Apache Spark integration for reading Vortex files

.. raw:: html

   <div class="api-links">
   <h3>API Documentation</h3>
   <ul>
     <li><a href="../../_static/vortex-jni/index.html">Vortex JNI API</a> - Core Java bindings for Vortex</li>
     <li><a href="../../_static/vortex-spark/index.html">Vortex Spark API</a> - Apache Spark integration</li>
   </ul>
   </div>

Installation
------------

The Java API can be included in your project using Gradle or Maven. Please refer to the main documentation for detailed installation instructions.


Compatibility
-------------

The Java bindings are supported on the following architectures:

* x86_64 Linux
* ARM64 Linux
* Apple Silicon macOS

They support any Linux distribution with a GLIBC version >= 2.31. This includes

* Amazon Linux 2022 or newer
* Ubuntu 20.04 or newer


Core API
--------

The main entry points of the Vortex Java API are:

* ``Session`` — the top-level handle that owns data sources and writers.
* ``DataSource`` — opens a Vortex file, glob, or table via ``DataSource.open(session, uri)`` and
  exposes its schema and row count.
* ``Scan`` / ``Partition`` — a scan over a ``DataSource`` iterates its partitions, yielding
  Arrow-compatible batches.
* ``VortexWriter`` — writes arrays to a Vortex file.

See the `Vortex JNI API <../../_static/vortex-jni/index.html>`_ above for the full,
method-level reference.
