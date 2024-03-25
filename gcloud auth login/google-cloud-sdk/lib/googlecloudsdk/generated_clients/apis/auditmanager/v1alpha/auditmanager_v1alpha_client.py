"""Generated client library for auditmanager version v1alpha."""
# NOTE: This file is autogenerated and should not be edited by hand.

from __future__ import absolute_import

from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.auditmanager.v1alpha import auditmanager_v1alpha_messages as messages


class AuditmanagerV1alpha(base_api.BaseApiClient):
  """Generated client library for service auditmanager version v1alpha."""

  MESSAGES_MODULE = messages
  BASE_URL = 'https://auditmanager.googleapis.com/'
  MTLS_BASE_URL = 'https://auditmanager.mtls.googleapis.com/'

  _PACKAGE = 'auditmanager'
  _SCOPES = ['https://www.googleapis.com/auth/cloud-platform']
  _VERSION = 'v1alpha'
  _CLIENT_ID = 'CLIENT_ID'
  _CLIENT_SECRET = 'CLIENT_SECRET'
  _USER_AGENT = 'google-cloud-sdk'
  _CLIENT_CLASS_NAME = 'AuditmanagerV1alpha'
  _URL_VERSION = 'v1alpha'
  _API_KEY = None

  def __init__(self, url='', credentials=None,
               get_credentials=True, http=None, model=None,
               log_request=False, log_response=False,
               credentials_args=None, default_global_params=None,
               additional_http_headers=None, response_encoding=None):
    """Create a new auditmanager handle."""
    url = url or self.BASE_URL
    super(AuditmanagerV1alpha, self).__init__(
        url, credentials=credentials,
        get_credentials=get_credentials, http=http, model=model,
        log_request=log_request, log_response=log_response,
        credentials_args=credentials_args,
        default_global_params=default_global_params,
        additional_http_headers=additional_http_headers,
        response_encoding=response_encoding)
    self.folders_locations_auditReports = self.FoldersLocationsAuditReportsService(self)
    self.folders_locations_auditScopeReports = self.FoldersLocationsAuditScopeReportsService(self)
    self.folders_locations_operationDetails = self.FoldersLocationsOperationDetailsService(self)
    self.folders_locations_operationIds = self.FoldersLocationsOperationIdsService(self)
    self.folders_locations = self.FoldersLocationsService(self)
    self.folders = self.FoldersService(self)
    self.projects_locations_auditReports = self.ProjectsLocationsAuditReportsService(self)
    self.projects_locations_auditScopeReports = self.ProjectsLocationsAuditScopeReportsService(self)
    self.projects_locations_operationDetails = self.ProjectsLocationsOperationDetailsService(self)
    self.projects_locations_operationIds = self.ProjectsLocationsOperationIdsService(self)
    self.projects_locations_operations = self.ProjectsLocationsOperationsService(self)
    self.projects_locations = self.ProjectsLocationsService(self)
    self.projects = self.ProjectsService(self)

  class FoldersLocationsAuditReportsService(base_api.BaseApiService):
    """Service class for the folders_locations_auditReports resource."""

    _NAME = 'folders_locations_auditReports'

    def __init__(self, client):
      super(AuditmanagerV1alpha.FoldersLocationsAuditReportsService, self).__init__(client)
      self._upload_configs = {
          }

    def Generate(self, request, global_params=None):
      r"""Register the Audit Report generation requests and returns the OperationId using which the customer can track the report generation progress.

      Args:
        request: (AuditmanagerFoldersLocationsAuditReportsGenerateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
      config = self.GetMethodConfig('Generate')
      return self._RunMethod(
          config, request, global_params=global_params)

    Generate.method_config = lambda: base_api.ApiMethodInfo(
        flat_path='v1alpha/folders/{foldersId}/locations/{locationsId}/auditReports:generate',
        http_method='POST',
        method_id='auditmanager.folders.locations.auditReports.generate',
        ordered_params=['scope'],
        path_params=['scope'],
        query_params=[],
        relative_path='v1alpha/{+scope}/auditReports:generate',
        request_field='generateAuditReportRequest',
        request_type_name='AuditmanagerFoldersLocationsAuditReportsGenerateRequest',
        response_type_name='Operation',
        supports_download=False,
    )

  class FoldersLocationsAuditScopeReportsService(base_api.BaseApiService):
    """Service class for the folders_locations_auditScopeReports resource."""

    _NAME = 'folders_locations_auditScopeReports'

    def __init__(self, client):
      super(AuditmanagerV1alpha.FoldersLocationsAuditScopeReportsService, self).__init__(client)
      self._upload_configs = {
          }

    def Generate(self, request, global_params=None):
      r"""Generates a demo report highlighting different responsibilities (Google/Customer/ shared) required to be fulfilled for the customer's workload to be compliant with the given standard.

      Args:
        request: (AuditmanagerFoldersLocationsAuditScopeReportsGenerateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (AuditScopeReport) The response message.
      """
      config = self.GetMethodConfig('Generate')
      return self._RunMethod(
          config, request, global_params=global_params)

    Generate.method_config = lambda: base_api.ApiMethodInfo(
        flat_path='v1alpha/folders/{foldersId}/locations/{locationsId}/auditScopeReports:generate',
        http_method='POST',
        method_id='auditmanager.folders.locations.auditScopeReports.generate',
        ordered_params=['scope'],
        path_params=['scope'],
        query_params=[],
        relative_path='v1alpha/{+scope}/auditScopeReports:generate',
        request_field='generateAuditScopeReportRequest',
        request_type_name='AuditmanagerFoldersLocationsAuditScopeReportsGenerateRequest',
        response_type_name='AuditScopeReport',
        supports_download=False,
    )

  class FoldersLocationsOperationDetailsService(base_api.BaseApiService):
    """Service class for the folders_locations_operationDetails resource."""

    _NAME = 'folders_locations_operationDetails'

    def __init__(self, client):
      super(AuditmanagerV1alpha.FoldersLocationsOperationDetailsService, self).__init__(client)
      self._upload_configs = {
          }

    def Get(self, request, global_params=None):
      r"""Get details about generate audit report operation.

      Args:
        request: (AuditmanagerFoldersLocationsOperationDetailsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
      config = self.GetMethodConfig('Get')
      return self._RunMethod(
          config, request, global_params=global_params)

    Get.method_config = lambda: base_api.ApiMethodInfo(
        flat_path='v1alpha/folders/{foldersId}/locations/{locationsId}/operationDetails/{operationDetailsId}',
        http_method='GET',
        method_id='auditmanager.folders.locations.operationDetails.get',
        ordered_params=['name'],
        path_params=['name'],
        query_params=[],
        relative_path='v1alpha/{+name}',
        request_field='',
        request_type_name='AuditmanagerFoldersLocationsOperationDetailsGetRequest',
        response_type_name='Operation',
        supports_download=False,
    )

  class FoldersLocationsOperationIdsService(base_api.BaseApiService):
    """Service class for the folders_locations_operationIds resource."""

    _NAME = 'folders_locations_operationIds'

    def __init__(self, client):
      super(AuditmanagerV1alpha.FoldersLocationsOperationIdsService, self).__init__(client)
      self._upload_configs = {
          }

    def Get(self, request, global_params=None):
      r"""Get details about generate audit report operation.

      Args:
        request: (AuditmanagerFoldersLocationsOperationIdsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
      config = self.GetMethodConfig('Get')
      return self._RunMethod(
          config, request, global_params=global_params)

    Get.method_config = lambda: base_api.ApiMethodInfo(
        flat_path='v1alpha/folders/{foldersId}/locations/{locationsId}/operationIds/{operationIdsId}',
        http_method='GET',
        method_id='auditmanager.folders.locations.operationIds.get',
        ordered_params=['name'],
        path_params=['name'],
        query_params=[],
        relative_path='v1alpha/{+name}',
        request_field='',
        request_type_name='AuditmanagerFoldersLocationsOperationIdsGetRequest',
        response_type_name='Operation',
        supports_download=False,
    )

  class FoldersLocationsService(base_api.BaseApiService):
    """Service class for the folders_locations resource."""

    _NAME = 'folders_locations'

    def __init__(self, client):
      super(AuditmanagerV1alpha.FoldersLocationsService, self).__init__(client)
      self._upload_configs = {
          }

    def EnrollResource(self, request, global_params=None):
      r"""Enrolls the customer resource(folder/project) to the audit manager service by creating the audit managers P4SA in customers workload and granting required permissions to the P4SA. Please note that if enrollment request is made on the already enrolled workload then enrollment is executed overriding the existing set of destinations. As per https://google.aip.dev/127 recommendation, we are having multiple URI binding for Enroll API.

      Args:
        request: (AuditmanagerFoldersLocationsEnrollResourceRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Enrollment) The response message.
      """
      config = self.GetMethodConfig('EnrollResource')
      return self._RunMethod(
          config, request, global_params=global_params)

    EnrollResource.method_config = lambda: base_api.ApiMethodInfo(
        flat_path='v1alpha/folders/{foldersId}/locations/{locationsId}:enrollResource',
        http_method='POST',
        method_id='auditmanager.folders.locations.enrollResource',
        ordered_params=['scope'],
        path_params=['scope'],
        query_params=[],
        relative_path='v1alpha/{+scope}:enrollResource',
        request_field='enrollResourceRequest',
        request_type_name='AuditmanagerFoldersLocationsEnrollResourceRequest',
        response_type_name='Enrollment',
        supports_download=False,
    )

  class FoldersService(base_api.BaseApiService):
    """Service class for the folders resource."""

    _NAME = 'folders'

    def __init__(self, client):
      super(AuditmanagerV1alpha.FoldersService, self).__init__(client)
      self._upload_configs = {
          }

  class ProjectsLocationsAuditReportsService(base_api.BaseApiService):
    """Service class for the projects_locations_auditReports resource."""

    _NAME = 'projects_locations_auditReports'

    def __init__(self, client):
      super(AuditmanagerV1alpha.ProjectsLocationsAuditReportsService, self).__init__(client)
      self._upload_configs = {
          }

    def Generate(self, request, global_params=None):
      r"""Register the Audit Report generation requests and returns the OperationId using which the customer can track the report generation progress.

      Args:
        request: (AuditmanagerProjectsLocationsAuditReportsGenerateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
      config = self.GetMethodConfig('Generate')
      return self._RunMethod(
          config, request, global_params=global_params)

    Generate.method_config = lambda: base_api.ApiMethodInfo(
        flat_path='v1alpha/projects/{projectsId}/locations/{locationsId}/auditReports:generate',
        http_method='POST',
        method_id='auditmanager.projects.locations.auditReports.generate',
        ordered_params=['scope'],
        path_params=['scope'],
        query_params=[],
        relative_path='v1alpha/{+scope}/auditReports:generate',
        request_field='generateAuditReportRequest',
        request_type_name='AuditmanagerProjectsLocationsAuditReportsGenerateRequest',
        response_type_name='Operation',
        supports_download=False,
    )

  class ProjectsLocationsAuditScopeReportsService(base_api.BaseApiService):
    """Service class for the projects_locations_auditScopeReports resource."""

    _NAME = 'projects_locations_auditScopeReports'

    def __init__(self, client):
      super(AuditmanagerV1alpha.ProjectsLocationsAuditScopeReportsService, self).__init__(client)
      self._upload_configs = {
          }

    def Generate(self, request, global_params=None):
      r"""Generates a demo report highlighting different responsibilities (Google/Customer/ shared) required to be fulfilled for the customer's workload to be compliant with the given standard.

      Args:
        request: (AuditmanagerProjectsLocationsAuditScopeReportsGenerateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (AuditScopeReport) The response message.
      """
      config = self.GetMethodConfig('Generate')
      return self._RunMethod(
          config, request, global_params=global_params)

    Generate.method_config = lambda: base_api.ApiMethodInfo(
        flat_path='v1alpha/projects/{projectsId}/locations/{locationsId}/auditScopeReports:generate',
        http_method='POST',
        method_id='auditmanager.projects.locations.auditScopeReports.generate',
        ordered_params=['scope'],
        path_params=['scope'],
        query_params=[],
        relative_path='v1alpha/{+scope}/auditScopeReports:generate',
        request_field='generateAuditScopeReportRequest',
        request_type_name='AuditmanagerProjectsLocationsAuditScopeReportsGenerateRequest',
        response_type_name='AuditScopeReport',
        supports_download=False,
    )

  class ProjectsLocationsOperationDetailsService(base_api.BaseApiService):
    """Service class for the projects_locations_operationDetails resource."""

    _NAME = 'projects_locations_operationDetails'

    def __init__(self, client):
      super(AuditmanagerV1alpha.ProjectsLocationsOperationDetailsService, self).__init__(client)
      self._upload_configs = {
          }

    def Get(self, request, global_params=None):
      r"""Get details about generate audit report operation.

      Args:
        request: (AuditmanagerProjectsLocationsOperationDetailsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
      config = self.GetMethodConfig('Get')
      return self._RunMethod(
          config, request, global_params=global_params)

    Get.method_config = lambda: base_api.ApiMethodInfo(
        flat_path='v1alpha/projects/{projectsId}/locations/{locationsId}/operationDetails/{operationDetailsId}',
        http_method='GET',
        method_id='auditmanager.projects.locations.operationDetails.get',
        ordered_params=['name'],
        path_params=['name'],
        query_params=[],
        relative_path='v1alpha/{+name}',
        request_field='',
        request_type_name='AuditmanagerProjectsLocationsOperationDetailsGetRequest',
        response_type_name='Operation',
        supports_download=False,
    )

  class ProjectsLocationsOperationIdsService(base_api.BaseApiService):
    """Service class for the projects_locations_operationIds resource."""

    _NAME = 'projects_locations_operationIds'

    def __init__(self, client):
      super(AuditmanagerV1alpha.ProjectsLocationsOperationIdsService, self).__init__(client)
      self._upload_configs = {
          }

    def Get(self, request, global_params=None):
      r"""Get details about generate audit report operation.

      Args:
        request: (AuditmanagerProjectsLocationsOperationIdsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
      config = self.GetMethodConfig('Get')
      return self._RunMethod(
          config, request, global_params=global_params)

    Get.method_config = lambda: base_api.ApiMethodInfo(
        flat_path='v1alpha/projects/{projectsId}/locations/{locationsId}/operationIds/{operationIdsId}',
        http_method='GET',
        method_id='auditmanager.projects.locations.operationIds.get',
        ordered_params=['name'],
        path_params=['name'],
        query_params=[],
        relative_path='v1alpha/{+name}',
        request_field='',
        request_type_name='AuditmanagerProjectsLocationsOperationIdsGetRequest',
        response_type_name='Operation',
        supports_download=False,
    )

  class ProjectsLocationsOperationsService(base_api.BaseApiService):
    """Service class for the projects_locations_operations resource."""

    _NAME = 'projects_locations_operations'

    def __init__(self, client):
      super(AuditmanagerV1alpha.ProjectsLocationsOperationsService, self).__init__(client)
      self._upload_configs = {
          }

    def Cancel(self, request, global_params=None):
      r"""Starts asynchronous cancellation on a long-running operation. The server makes a best effort to cancel the operation, but success is not guaranteed. If the server doesn't support this method, it returns `google.rpc.Code.UNIMPLEMENTED`. Clients can use Operations.GetOperation or other methods to check whether the cancellation succeeded or whether the operation completed despite cancellation. On successful cancellation, the operation is not deleted; instead, it becomes an operation with an Operation.error value with a google.rpc.Status.code of 1, corresponding to `Code.CANCELLED`.

      Args:
        request: (AuditmanagerProjectsLocationsOperationsCancelRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Empty) The response message.
      """
      config = self.GetMethodConfig('Cancel')
      return self._RunMethod(
          config, request, global_params=global_params)

    Cancel.method_config = lambda: base_api.ApiMethodInfo(
        flat_path='v1alpha/projects/{projectsId}/locations/{locationsId}/operations/{operationsId}:cancel',
        http_method='POST',
        method_id='auditmanager.projects.locations.operations.cancel',
        ordered_params=['name'],
        path_params=['name'],
        query_params=[],
        relative_path='v1alpha/{+name}:cancel',
        request_field='cancelOperationRequest',
        request_type_name='AuditmanagerProjectsLocationsOperationsCancelRequest',
        response_type_name='Empty',
        supports_download=False,
    )

    def Delete(self, request, global_params=None):
      r"""Deletes a long-running operation. This method indicates that the client is no longer interested in the operation result. It does not cancel the operation. If the server doesn't support this method, it returns `google.rpc.Code.UNIMPLEMENTED`.

      Args:
        request: (AuditmanagerProjectsLocationsOperationsDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Empty) The response message.
      """
      config = self.GetMethodConfig('Delete')
      return self._RunMethod(
          config, request, global_params=global_params)

    Delete.method_config = lambda: base_api.ApiMethodInfo(
        flat_path='v1alpha/projects/{projectsId}/locations/{locationsId}/operations/{operationsId}',
        http_method='DELETE',
        method_id='auditmanager.projects.locations.operations.delete',
        ordered_params=['name'],
        path_params=['name'],
        query_params=[],
        relative_path='v1alpha/{+name}',
        request_field='',
        request_type_name='AuditmanagerProjectsLocationsOperationsDeleteRequest',
        response_type_name='Empty',
        supports_download=False,
    )

    def Get(self, request, global_params=None):
      r"""Gets the latest state of a long-running operation. Clients can use this method to poll the operation result at intervals as recommended by the API service.

      Args:
        request: (AuditmanagerProjectsLocationsOperationsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
      config = self.GetMethodConfig('Get')
      return self._RunMethod(
          config, request, global_params=global_params)

    Get.method_config = lambda: base_api.ApiMethodInfo(
        flat_path='v1alpha/projects/{projectsId}/locations/{locationsId}/operations/{operationsId}',
        http_method='GET',
        method_id='auditmanager.projects.locations.operations.get',
        ordered_params=['name'],
        path_params=['name'],
        query_params=[],
        relative_path='v1alpha/{+name}',
        request_field='',
        request_type_name='AuditmanagerProjectsLocationsOperationsGetRequest',
        response_type_name='Operation',
        supports_download=False,
    )

    def List(self, request, global_params=None):
      r"""Lists operations that match the specified filter in the request. If the server doesn't support this method, it returns `UNIMPLEMENTED`.

      Args:
        request: (AuditmanagerProjectsLocationsOperationsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListOperationsResponse) The response message.
      """
      config = self.GetMethodConfig('List')
      return self._RunMethod(
          config, request, global_params=global_params)

    List.method_config = lambda: base_api.ApiMethodInfo(
        flat_path='v1alpha/projects/{projectsId}/locations/{locationsId}/operations',
        http_method='GET',
        method_id='auditmanager.projects.locations.operations.list',
        ordered_params=['name'],
        path_params=['name'],
        query_params=['filter', 'pageSize', 'pageToken'],
        relative_path='v1alpha/{+name}/operations',
        request_field='',
        request_type_name='AuditmanagerProjectsLocationsOperationsListRequest',
        response_type_name='ListOperationsResponse',
        supports_download=False,
    )

  class ProjectsLocationsService(base_api.BaseApiService):
    """Service class for the projects_locations resource."""

    _NAME = 'projects_locations'

    def __init__(self, client):
      super(AuditmanagerV1alpha.ProjectsLocationsService, self).__init__(client)
      self._upload_configs = {
          }

    def EnrollResource(self, request, global_params=None):
      r"""Enrolls the customer resource(folder/project) to the audit manager service by creating the audit managers P4SA in customers workload and granting required permissions to the P4SA. Please note that if enrollment request is made on the already enrolled workload then enrollment is executed overriding the existing set of destinations. As per https://google.aip.dev/127 recommendation, we are having multiple URI binding for Enroll API.

      Args:
        request: (AuditmanagerProjectsLocationsEnrollResourceRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Enrollment) The response message.
      """
      config = self.GetMethodConfig('EnrollResource')
      return self._RunMethod(
          config, request, global_params=global_params)

    EnrollResource.method_config = lambda: base_api.ApiMethodInfo(
        flat_path='v1alpha/projects/{projectsId}/locations/{locationsId}:enrollResource',
        http_method='POST',
        method_id='auditmanager.projects.locations.enrollResource',
        ordered_params=['scope'],
        path_params=['scope'],
        query_params=[],
        relative_path='v1alpha/{+scope}:enrollResource',
        request_field='enrollResourceRequest',
        request_type_name='AuditmanagerProjectsLocationsEnrollResourceRequest',
        response_type_name='Enrollment',
        supports_download=False,
    )

    def Get(self, request, global_params=None):
      r"""Gets information about a location.

      Args:
        request: (AuditmanagerProjectsLocationsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Location) The response message.
      """
      config = self.GetMethodConfig('Get')
      return self._RunMethod(
          config, request, global_params=global_params)

    Get.method_config = lambda: base_api.ApiMethodInfo(
        flat_path='v1alpha/projects/{projectsId}/locations/{locationsId}',
        http_method='GET',
        method_id='auditmanager.projects.locations.get',
        ordered_params=['name'],
        path_params=['name'],
        query_params=[],
        relative_path='v1alpha/{+name}',
        request_field='',
        request_type_name='AuditmanagerProjectsLocationsGetRequest',
        response_type_name='Location',
        supports_download=False,
    )

    def List(self, request, global_params=None):
      r"""Lists information about the supported locations for this service.

      Args:
        request: (AuditmanagerProjectsLocationsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListLocationsResponse) The response message.
      """
      config = self.GetMethodConfig('List')
      return self._RunMethod(
          config, request, global_params=global_params)

    List.method_config = lambda: base_api.ApiMethodInfo(
        flat_path='v1alpha/projects/{projectsId}/locations',
        http_method='GET',
        method_id='auditmanager.projects.locations.list',
        ordered_params=['name'],
        path_params=['name'],
        query_params=['filter', 'pageSize', 'pageToken'],
        relative_path='v1alpha/{+name}/locations',
        request_field='',
        request_type_name='AuditmanagerProjectsLocationsListRequest',
        response_type_name='ListLocationsResponse',
        supports_download=False,
    )

  class ProjectsService(base_api.BaseApiService):
    """Service class for the projects resource."""

    _NAME = 'projects'

    def __init__(self, client):
      super(AuditmanagerV1alpha.ProjectsService, self).__init__(client)
      self._upload_configs = {
          }