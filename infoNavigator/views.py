import csv
import logging
from decimal import Decimal
from StringIO import StringIO

from dateutil import parser
from django.core.paginator import Paginator
from django.db.models.loading import get_model
from django.db import models
from django.http import HttpResponse, StreamingHttpResponse
from django.template import loader
from django.utils.functional import cached_property
from django.utils.lru_cache import lru_cache
from django.shortcuts import render
from django import http

logger = logging.getLogger('django.request')


# Create your views here.


# TODO: Not really sure if any of this is thread safe lol
# TODO: see if there's a way to override view rather than do this
class ViewCallableMixin():
    http_method_names = ['get', 'post', 'put', 'patch', 'delete', 'head', 'options', 'trace']

    def as_view(self):
        def view(request, *args, **kwargs):
            if hasattr(self, 'get') and not hasattr(self, 'head'):
                self.head = self.get
            return self.dispatch(request, *args, **kwargs)

        return view

    def dispatch(self, request, *args, **kwargs):
        # Try to dispatch to the right method; if a method doesn't exist,
        # defer to the error handler. Also defer to the error handler if the
        # request method isn't on the approved list.
        if request.method.lower() in self.http_method_names:
            handler = getattr(self, request.method.lower())
        else:
            handler = self.http_method_not_allowed

        return handler(request, *args, **kwargs)

    def http_method_not_allowed(self, request, *args, **kwargs):
        logger.warning('Method Not Allowed (%s): %s', request.method, request.path,
                       extra={
                           'status_code': 405,
                           'request': request
                       }
                       )
        return http.HttpResponseNotAllowed(self._allowed_methods())

    def _allowed_methods(self):
        return [m.upper() for m in self.http_method_names if hasattr(self, m)]


class RecordView(object, ViewCallableMixin):
    PAGE_SIZE = 25
    template_name = 'infoNavigator/default_record_template.html'
    choose_columns_template_name = 'infoNavigator/choose_columns_template.html'

    def boolean_conversion_function(raw_str):
        raw_str = raw_str.lower()
        if raw_str != 'true' and raw_str != 'false':
            raise ValueError('Boolean filter makes no sense')
        return True if raw_str == 'true' else False

    def null_boolean_conversion_function(raw_str):
        raw_str = raw_str.lower()
        try:
            return RecordView.boolean_conversion_function(raw_str)
        except ValueError:
            if raw_str == 'none':
                return None
        finally:
            raise ValueError('Null boolean filter makes no sense')

    # Mapping of field types to functions that convert from their string representation to their python equivalent
    # and the valid ways that they can be filtered
    # Used for user filter functionality

    FIELD_TYPE_MAPPING = {
        models.BigIntegerField: (lambda raw_str: int(raw_str), {'lt', 'gt', 'lte', 'gte', 'exact'}),
        models.BooleanField: (boolean_conversion_function, {'exact', }),
        models.CharField: (lambda raw_str: raw_str, {'exact', 'icontains', 'contains'}),
        models.DateField: (lambda raw_str: parser.parse(raw_str).date(), {'lt', 'gt', 'lte', 'gte', 'exact'}),
        models.DateTimeField: (lambda raw_str: parser.parse(raw_str), {'lt', 'gt', 'lte', 'gte', 'exact'}),
        models.DecimalField: (lambda raw_str: Decimal(raw_str), {'lt', 'gt', 'lte', 'gte', 'exact'}),
        models.EmailField: (lambda raw_str: raw_str, {'exact', 'icontains', 'contains'}),
        # If you're wondering why there's no "exact" for floats, it's because it would be very fickle
        models.FloatField: (lambda raw_str: float(raw_str), {'lt', 'gt', 'lte', 'gte'}),
        models.IntegerField: (lambda raw_str: int(raw_str), {'lt', 'gt', 'lte', 'gte', 'exact'}),
        models.NullBooleanField: (null_boolean_conversion_function, {'exact', }),
        models.PositiveIntegerField: (lambda raw_str: int(raw_str), {'lt', 'gt', 'lte', 'gte', 'exact'}),
        models.PositiveSmallIntegerField: (lambda raw_str: int(raw_str), {'lt', 'gt', 'lte', 'gte', 'exact'}),
        models.SlugField: (lambda raw_str: raw_str, {'exact', 'icontains', 'contains'}),
        models.SmallIntegerField: (lambda raw_str: int(raw_str), {'lt', 'gt', 'lte', 'gte', 'exact'}),
        models.TextField: (lambda raw_str: raw_str, {'exact', 'icontains', 'contains'}),
        models.TimeField: (lambda raw_str: parser.parse(raw_str).time(), {'lt', 'gt', 'lte', 'gte', 'exact'}),
        models.URLField: (lambda raw_str: raw_str, {'exact', 'icontains', 'contains'}),
    }

    def __init__(self, model, **kwargs):
        """
        Provide a model and a table will be generated for you by
        going through the model, following its foreign keys to its related models,
        following their foreign keys, etc.
        :param model:
        :param kwargs:
        :return:
        """
        self.model = model

        super(RecordView, self).__init__(**kwargs)

    @lru_cache(maxsize=1000)
    def _get_model_fields(self, app_label, model_name, foreign_fields):
        """
        Returns django fields for a model. If foreign_fields is True it returns those fields (other than one_to_many)
        that link a model to other models. If foreign_fields is False it returns those fields that are not a link to
        other models
        :param app_label: label of django app
        :param model_name: name of model in django app
        :param foreign_fields: toggles whether the method will return fields that are related fields or fields that are
        not related fields
        :return:
        """
        # the reason for passing model_name and app_label rather
        # than just model is so we can take advantage of lru_cache
        model = get_model(app_label, model_name=model_name)
        return [field for field in model._meta.get_fields()
                if (foreign_fields and field.is_relation and not field.one_to_many) or (
                    not foreign_fields and not field.is_relation)]

    @cached_property
    def _all_record_keys(self):
        keys = set()
        self._get_all_record_keys_helper(keys, self.model._meta.app_label,
                                         self.model._meta.object_name, set())
        return keys

    def _get_all_record_keys_helper(self, keys, curr_app_label, curr_model_label, visited):
        if (curr_app_label, curr_model_label) in visited:
            return
        visited.add((curr_app_label, curr_model_label))

        fields_non_foreign = self._get_model_fields(curr_app_label, curr_model_label, False)
        fields_foreign = self._get_model_fields(curr_app_label, curr_model_label, True)

        for field in fields_non_foreign:
            keys.add(','.join([curr_app_label, curr_model_label, field.name]))

        for field in fields_foreign:
            if field.many_to_many:
                curr_model = get_model(curr_app_label, model_name=curr_model_label)
                if hasattr(curr_model, field.name):
                    keys.add(','.join([curr_app_label, curr_model_label, field.name]))
            if field.many_to_one:
                self._get_all_record_keys_helper(keys, field.related_model._meta.app_label,
                                                 field.related_model._meta.object_name, visited)

    def _get_records(self, queryset):
        records = [self._get_record(model_instance) for model_instance in queryset]
        keys = self._all_record_keys
        for row in records:
            for key in keys:
                if key not in row:
                    row[key] = 'N/A'

        # sort the keys alphabetically and then change our records accordingly
        keys = sorted(list(keys))
        for i in range(0, len(records)):
            records[i] = [records[i][key] for key in keys]

        return records, keys

    def _get_record(self, model_instance):
        """
        Returns a record_dict for a model instance that represents it and the attributes of all its related instances
        :param model_instance:
        :return:
        """
        record_dict = {}
        visited = set()
        self._get_record_helper(model_instance._meta.app_label, model_instance._meta.object_name, record_dict,
                                visited,
                                model_instance)
        return record_dict

    def _get_field_sort_key(self, field):
        """
        Key function for sorting normal fields first, many_to_many second, and the rest last
        :param field:
        :return:
        """
        if not field.is_relation:
            return -1
        return 0 if field.many_to_many else 1

    def _get_record_helper(self, app_label, model_name, record_dict, visited, instance):
        """
        Recursive helper function called from _get_record to do all the real work
        :param app_label:
        :param model_name:
        :param record_dict:
        :param visited:
        :param instance:
        :return:
        """
        # if the model has already been visited, bail out to avoid circular linkage (e.g. A -> B -> A)
        # TODO: Explore behavior of this solution more thoroughly
        if (app_label, model_name) in visited:
            return
        visited.add((app_label, model_name))
        for field in self._get_model_fields(app_label, model_name, False):
            record_dict[','.join([app_label, model_name, field.name])] = unicode((getattr(instance, field.name)))

        for field in sorted(self._get_model_fields(app_label, model_name, True), key=self._get_field_sort_key):
            if field.many_to_many:
                # Not sure if I like the way this behaves when the model
                # referenced via ManyToMany has its own related fields
                if hasattr(instance, field.name):
                    record_dict[','.join([app_label, model_name, field.name])] = ','.join(
                        [unicode(instance) for instance in getattr(instance, field.name).all()]
                    )
            if field.many_to_one:
                if hasattr(instance, field.name) and getattr(instance, field.name) is not None:
                    self._get_record_helper(field.related_model._meta.app_label, field.related_model._meta.object_name,
                                            record_dict, visited, getattr(instance, field.name))

    # TODO: This has undefined behavior for M2M fields
    # It would be a lot better to have LRU cache on _find_order_helper, but it has a bug with lru_cache, so this
    # is a stop-gap of sorts
    @lru_cache(1000)
    def _find_sort_order_join(self, sort_key):
        if sort_key[0] == '-':
            sort_char = '-'
            sort_key = sort_key[1:]
        else:
            sort_char = ''

        return sort_char + self._find_order_helper(sort_key, self.model._meta.app_label,
                                                   self.model._meta.object_name, '', set())[0]

    # It would be a lot better to have LRU cache on _find_order_helper, but it has a bug with lru_cache, so this
    # is a stop-gap of sorts
    @lru_cache(1000)
    def _find_filter_join(self, filter_key, specifier):
        filter_join_string, field = self._find_order_helper(filter_key,
                                                            self.model._meta.app_label,
                                                            self.model._meta.object_name, '', set())

        if type(field) not in self.FIELD_TYPE_MAPPING:
            raise ValueError('Filtering on this field is not supported')

        if specifier not in self.FIELD_TYPE_MAPPING[type(field)][1]:
            raise ValueError('Filtering on this field with that specifier is not supported')

        return filter_join_string + '__' + specifier, field

    # TODO: Implement LRU cache here
    def _find_order_helper(self, sort_key, curr_app_label, curr_model_name, order_string, visited):
        if (curr_app_label, curr_model_name) in visited:
            return None
        visited.add((curr_app_label, curr_model_name))
        app_label, model_name, field_name = sort_key.split(',')
        if curr_app_label == app_label and curr_model_name == model_name:
            return order_string + field_name, \
                   get_model(app_label, model_name=model_name)._meta.get_field(field_name)

        foreign_fields = self._get_model_fields(curr_app_label, curr_model_name, True)
        for field in foreign_fields:
            order_helper_result = self._find_order_helper(sort_key,
                                                          field.related_model._meta.app_label,
                                                          field.related_model._meta.object_name,
                                                          order_string + field.name + '__', visited)
            if order_helper_result is not None:
                return order_helper_result

        return None

    def _generate_filter_object(self, field, raw_string):
        # TODO: Account for the case where they want values that are null. In other words field__exact=None. currently no way to do this.
        return self.FIELD_TYPE_MAPPING[type(field)][0](raw_string)

    def _apply_sort_keys(self, request, queryset):
        sort_keys = request.GET['sort_keys'].split(':') if 'sort_keys' in request.GET else []
        sort_keys = [sort_key for sort_key in sort_keys if len(sort_key) != 0]
        if len(sort_keys) > 0:
            queryset = queryset.order_by(*[self._find_sort_order_join(key) for key in sort_keys])
        return queryset

    def _get_filter_keys_and_values(self, request):
        filter_keys_and_values = {}
        # format of filter_keys_and_values:
        # "app_name,model_name,field_name~~~thingtofilteron~~~specifier:::app_name,model_name,field_name~~~thingtofilteron~~~specifier"
        if 'filters' in request.GET:
            for key_value_string in request.GET['filters'].split(':::'):
                if len(key_value_string) > 0:
                    # key is the field to be sorted on, value is the value to match, specifier
                    # is gt, lt, or exact
                    key, value, specifier = key_value_string.split('~~~')
                    if key in filter_keys_and_values:
                        filter_keys_and_values[key].append({'value': value,
                                                            'specifier': specifier})
                    else:
                        filter_keys_and_values[key] = [{'value': value,
                                                        'specifier': specifier}]
        return filter_keys_and_values

    def _apply_filter_keys(self, request, queryset):
        filter_keys_and_values = self._get_filter_keys_and_values(request)
        for key in filter_keys_and_values:
            for value in filter_keys_and_values[key]:
                filter_join_string, filter_field = self._find_filter_join(key, value['specifier'])

                filter_object = self._generate_filter_object(filter_field,
                                                             value['value'])
                queryset = queryset.filter(**{filter_join_string: filter_object})

        return queryset

    def _combine_keys_with_filters(self, request, keys):
        filter_keys_and_values = self._get_filter_keys_and_values(request)
        keys_and_filters = []
        for i in xrange(len(keys)):
            key_value = keys[i]
            filters_list = []
            if key_value in filter_keys_and_values:
                filter_values = filter_keys_and_values[key_value]
                for filter_value in filter_values:
                    filters_list.append({"value": filter_value['value'],
                                         "specifier": filter_value['specifier']})
            filter_field = self._find_order_helper(key_value,
                                                   self.model._meta.app_label,
                                                   self.model._meta.object_name, '', set())[1]
            if type(filter_field) not in self.FIELD_TYPE_MAPPING:
                filters_list = None
                possible_specifiers = None
            else:
                possible_specifiers = sorted(list(self.FIELD_TYPE_MAPPING[type(filter_field)][1]))
                # add one more blank filter with a default empty string so the user can add a filter
                filters_list.append({"value": "",
                                     "specifier": possible_specifiers[0]})
            keys_and_filters.append({
                "key": key_value,
                "filters": filters_list,
                "possible_specifiers": possible_specifiers
            })
        return keys_and_filters

    def _narrow_to_chosen_columns(self, request, records, keys):
        if 'chosen_columns' in request.GET:
            index_of_chosen_columns = []
            chosen_columns = set(request.GET['chosen_columns'].split('~'))
            for i, column_name in enumerate(keys):
                if column_name in chosen_columns:
                    index_of_chosen_columns.append(i)

            keys = [keys[i] for i in index_of_chosen_columns]

            for i in xrange(len(records)):
                current_record = records[i]
                new_record = []
                for j in index_of_chosen_columns:
                    new_record.append(current_record[j])
                records[i] = new_record

            return records, keys
        else:
            return None, None

    # TODO: Only use the request object in here so the methods are more portable
    def get_html_content(self, request):
        # - is descending order, absence is ascending
        # format of sort keys: "app_name,model_name,field_name:-app_name,model_name,field_name"
        queryset = self.model.objects.all()
        queryset = self._apply_sort_keys(request, queryset)
        try:
            queryset = self._apply_filter_keys(request, queryset)
        except ValueError:
            return 'Filter value bad'

        page_number = request.GET['pg'] if 'pg' in request.GET else 1
        paginator = Paginator(queryset, self.PAGE_SIZE)
        page = paginator.page(page_number)
        records, keys = self._get_records(page.object_list)
        records, keys = self._narrow_to_chosen_columns(request, records, keys)

        model_key = ','.join([self.model._meta.app_label,
                              self.model._meta.object_name])
        if records is None or keys is None:  # indicates that no columns have been chosen
            return loader.render_to_string(self.choose_columns_template_name,
                                           {'possible_keys': sorted(list(self._all_record_keys)),
                                            'base_url': request.path,
                                            'model': model_key,
                                            'modelApp': self.model._meta.app_label,
                                            'modelName': self.model._meta.object_name})

        keys_and_filters = self._combine_keys_with_filters(request, keys)

        return loader.render_to_string(self.template_name,
                                       {'records': records,
                                        'record_keys_and_filters': keys_and_filters,
                                        'page': page,
                                        'paginator': paginator,
                                        'sort_keys': request.GET['sort_keys'] if 'sort_keys' in request.GET else '',
                                        'base_url': request.path,
                                        'model': model_key,
                                        'filters': request.GET['filters'] if 'filters' in request.GET else '',
                                        'filter_param_string': request.GET[
                                            'filters'] if 'filters' in request.GET else '',
                                        'chosen_columns': request.GET['chosen_columns'],
                                        # Two below allow record views to be distinguished if there are multiple on screen
                                        'modelApp': self.model._meta.app_label,
                                        'modelName': self.model._meta.object_name})

    def _get_csv_iterator(self, queryset, request):
        NUMBER_RECORDS_PER_QUERY = 10 ** 3
        paginator = Paginator(queryset, NUMBER_RECORDS_PER_QUERY)
        strio = StringIO()
        writer = csv.writer(strio)
        dummy_records, selected_keys = self._narrow_to_chosen_columns(request,
                                                                      [],
                                                                      sorted(list(self._all_record_keys)))
        writer.writerow(selected_keys)
        yield strio.getvalue()
        strio.close()
        for i in xrange(1, paginator.num_pages + 1):
            strio = StringIO()
            writer = csv.writer(strio)
            page = paginator.page(i)
            records, keys = self._get_records(page.object_list)
            records, keys = self._narrow_to_chosen_columns(request, records, keys)
            for record in records:
                for j in xrange(len(record)):
                    record[j] = record[j].encode('utf-8')
            writer.writerows(records)
            yield strio.getvalue()
            strio.close()

    def get_csv_response(self, request):
        # - is descending order, absence is ascending
        # format of sort keys: "app_name,model_name,field_name:-app_name,model_name,field_name"
        queryset = self.model.objects.all()
        queryset = self._apply_sort_keys(request, queryset)
        try:
            queryset = self._apply_filter_keys(request, queryset)
        except ValueError:
            return 'Filter value bad'

        csv_iterator = self._get_csv_iterator(queryset, request)

        response = StreamingHttpResponse(csv_iterator, content_type='text/csv')
        response['Content-Disposition'] = 'attachment; filename="{}.csv"' \
            .format(','.join([self.model._meta.app_label,
                              self.model._meta.object_name]))
        return response

    def get(self, request):
        rendered_template = self.get_html_content(request)

        return render(request, 'infoNavigator/template_wrapper.html', {'rendered_template': rendered_template})


class RecordsView(object, ViewCallableMixin):
    template_name = 'infoNavigator/default_records_template.html'

    def __init__(self, *models, **kwargs):
        """
        Provide models and a record_view (if you have one that you want to override with)
        and an information listing for all those models will be generated for you by going through each model,
        following its foreign keys to its related models, following their foreign keys, etc. If you want to override
        which record view is used for a particular model, pass in the model as an iterable
        with the model first and the special record view second(e.g. (models.Accident, SpecialRecordView)
        :param record_view: Class implementing __init__ that takes a model class, get_html_content which when passed a request
        with params 'sort_keys', 'pg' returns a rendered string. See infoNavigator.views.RecordView for an example
        :param models: An arbitrary number of models for the RecordsView
        :param kwargs:
        :return:
        """
        self.models = models
        self.record_views = {}
        if 'record_view' in kwargs:
            record_view = kwargs['record_view']
        else:
            record_view = RecordView
        for model in self.models:
            if isinstance(model, list) or isinstance(model, tuple):
                model_key = ','.join([model[0]._meta.app_label, model[0]._meta.object_name])
                self.record_views[model_key] = model[1](model[0])
            else:
                model_key = ','.join([model._meta.app_label, model._meta.object_name])
                self.record_views[model_key] = record_view(model)

        super(RecordsView, self).__init__(**kwargs)

    def get(self, request):
        if 'model' not in request.GET:
            return render(request, self.template_name, {'record_rendered_template': 'No model chosen',
                                                        'model_keys': self.record_views.keys()})
        record_view = self.record_views[request.GET['model']]
        if 'csv' in request.GET:
            return record_view.get_csv_response(request)
        record_rendered_template = record_view.get_html_content(request)
        return render(request, self.template_name, {'record_rendered_template': record_rendered_template,
                                                    'model_keys': self.record_views.keys()})
